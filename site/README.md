# Site

This is the Next.js + Tailwind frontend for the tool embedding registry.

## Run

```bash
cd site
npm install
npm run dev
```

The app reads generated registry artifacts from `../registry/generated/`, so run
the registry builder first:

```bash
python3 scripts/build_registry.py
```
