import requests, re
r = requests.get('https://www.myscheme.gov.in/find-scheme', headers={'User-Agent': 'Mozilla/5.0'})
print('HTML len:', len(r.text))
jsons = re.findall(r'_next/data/[^/]+/find-scheme.json', r.text)
if jsons:
    print('Next.js data endpoint:', jsons[0])
    # fetch it
    d = requests.get(f"https://www.myscheme.gov.in/{jsons[0]}", headers={'User-Agent': 'Mozilla/5.0'}).json()
    print("Keys:", d.keys())
