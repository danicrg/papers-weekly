# ML Papers Weekly (free signals)

Twitter removed their free API, which sucks, so I will use Reddit, HackerNews and Bluesky.

Ranks last-7-day arXiv ML papers by free social/citation signals:
- Reddit (OAuth client credentials)
- Hacker News (Algolia)
- Bluesky (public AppView)

## Setup

1) Create a Reddit app at https://www.reddit.com/prefs/apps  
   - Type: "script" or "installed app"
   - Note the **client id** and **secret**

2) In your GitHub repo → Settings → Secrets and variables → Actions:
   - Add `REDDIT_CLIENT_ID`
   - Add `REDDIT_SECRET`

3) Enable GitHub Pages (or Vercel/Netlify) to serve `public/`.

4) Run locally:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export REDDIT_CLIENT_ID=xxx
export REDDIT_SECRET=yyy
python scripts/build_weekly.py