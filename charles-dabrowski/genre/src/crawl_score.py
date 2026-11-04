"""8th class: film scores. Deezer genre 173 (Films/Games) seeded from film composers + score playlists + the Film Scores radio.
Chart seeds are skipped on purpose (they are pop songs that appear in films)."""
import csv, random, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import scrape_deezer as sd

GID, NAME = 173, "score"
COMPOSERS = ["Hans Zimmer", "John Williams", "Ennio Morricone", "Howard Shore", "James Horner", "Alan Silvestri", "Danny Elfman",
             "Thomas Newman", "Alexandre Desplat", "James Newton Howard", "Ludwig Göransson", "Hildur Guðnadóttir", "Ramin Djawadi",
             "Michael Giacchino", "Jerry Goldsmith", "Bernard Herrmann", "John Barry", "Joe Hisaishi", "Nino Rota", "Vangelis",
             "Clint Mansell", "Johann Johannsson", "Max Richter", "Trent Reznor & Atticus Ross", "Hiroyuki Sawano", "Yann Tiersen",
             "Gustavo Santaolalla", "Daniel Pemberton", "Henry Jackman", "Harry Gregson-Williams", "John Powell", "Steve Jablonsky",
             "Brian Tyler", "Patrick Doyle", "Dario Marianelli", "Rachel Portman", "Carter Burwell", "Marco Beltrami", "Tyler Bates",
             "Lorne Balfe", "Benjamin Wallfisch", "Nicholas Britell", "Mica Levi", "Jonny Greenwood", "Bear McCreary", "Junkie XL",
             "Craig Armstrong", "Elliot Goldenthal", "Basil Poledouris", "Maurice Jarre", "Lalo Schifrin", "Angelo Badalamenti",
             "Philip Glass", "Thomas Bergersen", "Two Steps From Hell", "Audiomachine", "Ólafur Arnalds", "Dustin O'Halloran",
             "Nobuo Uematsu", "Koji Kondo", "Jeremy Soule", "Austin Wintory", "Gareth Coker", "Mick Gordon", "Jesper Kyd"]
PLAYLISTS = {"Film Classics", "Epic Soundtracks", "Fantasy Mode", "Cinematic Zen", "Cinematic Classical", "Soundtrack Classics",
             "League Of Legends + Arcane Soundtrack"}

def seed_artists(gid):
    ids = {}
    for name in COMPOSERS:
        d = sd.get("/search/artist", q=name, limit=1)
        if d and d.get("data"):
            ids[d["data"][0]["id"]] = d["data"][0]["name"]
    ed = sd.get(f"/editorial/{gid}/charts") or {}
    for p in ed.get("playlists", {}).get("data", []):
        if p["title"] in PLAYLISTS:
            for t in sd.get_all(f"/playlist/{p['id']}/tracks", max_items=200):
                ids[t["artist"]["id"]] = t["artist"]["name"]
    for r in (sd.get(f"/genre/{gid}/radios") or {}).get("data", []):
        if r["title"] == "Film Scores":
            for t in sd.get_all(f"/radio/{r['id']}/tracks", max_items=100):
                ids[t["artist"]["id"]] = t["artist"]["name"]
    return ids

sd.seed_artists = seed_artists
log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
rows = sd.crawl_genre(NAME, GID, target=1600, max_artists=600, per_artist=12, per_album=4, log=log)

existing = {r["artist_id"] for r in csv.DictReader(open(sd.ROOT / "tracks.csv"))}
rows = [r for r in rows if str(r["artist_id"]) not in existing]
log(f"{len(rows)} score tracks after removing artists already in other genres")
with open(sd.ROOT / "tracks_score.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

ok = 0
with ThreadPoolExecutor(8) as ex:
    for good in ex.map(lambda r: sd.download(r, sd.ROOT / "audio"), rows):
        ok += good
log(f"download: {ok}/{len(rows)}")
# refresh expired links once
miss = [r for r in rows if not (sd.ROOT / "audio" / NAME / f"{r['track_id']}.mp3").exists()]
for r in miss:
    t = sd.get(f"/track/{r['track_id']}")
    if t and t.get("preview"):
        r["preview"] = t["preview"]; ok += sd.download(r, sd.ROOT / "audio")
log(f"after refresh: {ok}/{len(rows)}")
# append to the master list
with open(sd.ROOT / "tracks.csv", "a", newline="") as f:
    csv.DictWriter(f, fieldnames=list(rows[0].keys())).writerows(rows)
log("SCORE CRAWL DONE")
