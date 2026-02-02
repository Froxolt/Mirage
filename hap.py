import asyncio
import aiohttp
import aiosqlite
import time
import math
import hashlib
import os
import re
import threading
from collections import defaultdict, Counter
from urllib.parse import urljoin, urlparse, parse_qs
from bs4 import BeautifulSoup
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.robotparser as robotparser


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = os.path.join(BASE_DIR, "public")
DB_FILE = os.path.join(BASE_DIR, "engine.db")

PORT = 8080
USER_AGENT = "MirageBot/1.0"

START_URLS = [
    "https://en.wikipedia.org/wiki/Main_Page",
    "https://neocities.org/",
]

CRAWLER_WORKERS = 8
DOMAIN_DELAY = 2
QUEUE_LIMIT = 20_000

# Search
WORD_RE = re.compile(r"[a-zA-Z0-9]{2,}")
k1, b = 1.5, 0.75
SNIPPET_LEN = 160

# PageRank
DAMPING = 0.85
MAX_ITER = 15
PR_INTERVAL = 60  # seconds

os.makedirs(PUBLIC_DIR, exist_ok=True)


SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS pages (
    url TEXT PRIMARY KEY,
    text TEXT,
    length INTEGER,
    hash TEXT,
    pagerank REAL DEFAULT 1.0
);

CREATE TABLE IF NOT EXISTS terms (
    term TEXT,
    url TEXT,
    freq INTEGER
);

CREATE TABLE IF NOT EXISTS links (
    src TEXT,
    dst TEXT
);

CREATE INDEX IF NOT EXISTS idx_terms_term ON terms(term);
CREATE INDEX IF NOT EXISTS idx_links_src ON links(src);
"""


def tokenize(text):
    return WORD_RE.findall(text.lower())

def normalize(base, link):
    try:
        u = urljoin(base, link)
        p = urlparse(u)
        if p.scheme not in ("http", "https"):
            return None
        return f"{p.scheme}://{p.netloc}{p.path.rstrip('/')}"
    except Exception:
        return None

def hash_text(text):
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()


class Politeness:
    def __init__(self):
        self.robots = {}
        self.last_fetch = defaultdict(float)

    async def allowed(self, url):
        base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        if base not in self.robots:
            rp = robotparser.RobotFileParser()
            rp.set_url(base + "/robots.txt")

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, rp.read)

            self.robots[base] = rp
        return self.robots[base].can_fetch(USER_AGENT, url)


    async def wait(self, url):
        domain = urlparse(url).netloc
        delta = time.time() - self.last_fetch[domain]
        if delta < DOMAIN_DELAY:
            await asyncio.sleep(DOMAIN_DELAY - delta)
        self.last_fetch[domain] = time.time()

politeness = Politeness()


def extract_main_text(soup):
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    blocks = soup.find_all(["article", "main", "div"])
    if not blocks:
        return ""

    def score(el):
        t = el.get_text(" ", strip=True)
        return len(t) + t.count(".") * 20

    best = max(blocks, key=score)
    return re.sub(r"\s+", " ", best.get_text(" ", strip=True))


seen_urls = set()
seen_lock = asyncio.Lock()

async def crawl_worker(queue):
    async with aiohttp.ClientSession(headers={"User-Agent": USER_AGENT}) as session:
        async with aiosqlite.connect(DB_FILE) as db:
            while True:
                url = await queue.get()

                async with seen_lock:
                    if url in seen_urls:
                        queue.task_done()
                        continue
                    seen_urls.add(url)

                if not await politeness.allowed(url):
                    queue.task_done()
                    continue

                await politeness.wait(url)

                try:
                    async with session.get(url, timeout=15) as r:
                        if "text/html" not in r.headers.get("Content-Type", ""):
                            queue.task_done()
                            continue
                        html = await r.text()
                except Exception:
                    queue.task_done()
                    continue

                soup = BeautifulSoup(html, "lxml")
                text = extract_main_text(soup)
                tokens = tokenize(text)
                if not tokens:
                    queue.task_done()
                    continue

                await db.execute(
                    "INSERT OR REPLACE INTO pages VALUES (?,?,?,?,COALESCE((SELECT pagerank FROM pages WHERE url=?),1.0))",
                    (url, text, len(tokens), hash_text(text), url)
                )

                await db.execute("DELETE FROM terms WHERE url=?", (url,))
                await db.executemany(
                    "INSERT INTO terms VALUES (?,?,?)",
                    [(t, url, f) for t, f in Counter(tokens).items()]
                )

                links = set()
                for a in soup.select("a[href]"):
                    link = normalize(url, a["href"])
                    if link:
                        links.add(link)

                await db.executemany(
                    "INSERT OR IGNORE INTO links VALUES (?,?)",
                    [(url, l) for l in links]
                )

                async with seen_lock:
                    for l in links:
                        if l not in seen_urls and queue.qsize() < QUEUE_LIMIT:
                            queue.put_nowait(l)

                await db.commit()
                queue.task_done()


async def pagerank_loop():
    while True:
        async with aiosqlite.connect(DB_FILE) as db:
            cur = await db.execute("SELECT url FROM pages")
            urls = [r[0] for r in await cur.fetchall()]
            N = len(urls)
            if N == 0:
                await asyncio.sleep(PR_INTERVAL)
                continue

            pr = {u: 1 / N for u in urls}
            adj = defaultdict(set)

            cur = await db.execute("SELECT src,dst FROM links")
            for src, dst in await cur.fetchall():
                if src in pr and dst in pr:
                    adj[src].add(dst)

            for _ in range(MAX_ITER):
                new_pr = {}
                for u in urls:
                    rank_sum = sum(pr[v] / len(adj[v]) for v in adj if u in adj[v])
                    new_pr[u] = (1 - DAMPING) / N + DAMPING * rank_sum
                pr = new_pr

            await db.executemany(
                "UPDATE pages SET pagerank=? WHERE url=?",
                [(score, url) for url, score in pr.items()]
            )
            await db.commit()

        await asyncio.sleep(PR_INTERVAL)


async def search(query, limit=20):
    terms = tokenize(query)
    if not terms:
        return []

    async with aiosqlite.connect(DB_FILE) as db:
        cur = await db.execute("SELECT COUNT(*), AVG(length) FROM pages")
        N, avg_len = await cur.fetchone()
        N = N or 1
        avg_len = avg_len or 1

        scores = defaultdict(float)
        snippets = {}

        for t in terms:
            cur = await db.execute("""
                SELECT p.url, p.length, p.pagerank, t.freq, p.text
                FROM terms t JOIN pages p ON p.url=t.url
                WHERE t.term=?
            """, (t,))
            rows = await cur.fetchall()
            df = len(rows)
            if not df:
                continue

            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

            for url, length, pr, freq, text in rows:
                score = idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * length / avg_len))
                scores[url] += score * pr

                if url not in snippets:
                    pos = text.lower().find(t)
                    if pos != -1:
                        s = max(0, pos - SNIPPET_LEN // 2)
                        e = min(len(text), pos + SNIPPET_LEN // 2)
                        snippet = re.sub(
                            f"(?i)({re.escape(t)})",
                            r"<b>\1</b>",
                            text[s:e]
                        )
                        snippets[url] = snippet

        return sorted(
            [(u, s, snippets.get(u, "")) for u, s in scores.items()],
            key=lambda x: x[1],
            reverse=True
        )[:limit]


class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        return os.path.join(PUBLIC_DIR, path.lstrip("/"))

    def do_GET(self):
        if self.path.startswith("/search"):
            qs = parse_qs(urlparse(self.path).query)
            q = qs.get("q", [""])[0]
            results = loop.run_until_complete(search(q))

            with open(os.path.join(PUBLIC_DIR, "search.html"), encoding="utf-8") as f:
                html = f.read()

            out = ""
            for url, score, snippet in results:
                out += f"<div><a href='{url}'>{url}</a><p>{snippet}</p></div>"

            html = html.replace("{{RESULTS}}", out or "<p>No results.</p>")
            self.send_response(200)
            self.end_headers()
            self.wfile.write(html.encode())
        else:
            super().do_GET()

def run_server():
    HTTPServer(("", PORT), Handler).serve_forever()


async def main():
    async with aiosqlite.connect(DB_FILE) as db:
        await db.executescript(SCHEMA)
        await db.commit()

    queue = asyncio.Queue()

    for u in START_URLS:
        queue.put_nowait(u)

    workers = [asyncio.create_task(crawl_worker(queue)) for _ in range(CRAWLER_WORKERS)]
    pr_task = asyncio.create_task(pagerank_loop())

    await asyncio.gather(*workers, pr_task)


loop = asyncio.new_event_loop()
threading.Thread(target=run_server, daemon=True).start()
print(f"Mirage running at http://localhost:{PORT}")

loop.run_until_complete(main())
