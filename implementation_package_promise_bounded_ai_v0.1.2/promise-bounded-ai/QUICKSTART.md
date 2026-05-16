# Quickstart

## For a human user

1. Open `site/index.html`.
2. Choose **Use the Navigator**.
3. Name one thing that is actually present.
4. Walk only as far as needed.
5. Copy the final next move or short summary.
6. Use that cleaner context before asking AI or acting.

## For an AI user

1. Use the Navigator to shape the situation.
2. Copy the short summary.
3. Paste it into any AI with this instruction:

> Use this context to reduce distortion. Do not treat any fragment as the whole. Preserve boundaries. Do not force closure. Help identify one truthful next move.

## For a builder

1. Read `ARCHITECTURE.md`.
2. Inspect `/schemas`.
3. Open `site/promise-builder.html`.
4. Test the examples in `/examples`.
5. Use the schemas to validate whether an agent action is promise-bounded.

## Do not start with autonomy

Do not connect tools, email, payments, calendars, file systems, or external APIs until the promise, boundary, cost, trace, and repair objects are explicit and validated.
