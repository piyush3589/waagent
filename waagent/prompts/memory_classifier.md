# Memory classifier (v1)

Decide whether a user message contains a durable, user-specific fact worth
storing long-term, and if so, its kind:

- `preference` - language, channel, contact-time, do-not-call
- `fact`       - name, city, account/product details (non-PII)
- `event`      - an order/ticket/complaint that may recur
- `sentiment`  - a durable feeling about the brand

Store only durable + user-specific + non-PII content. Never store greetings,
one-off questions, or anything containing phone/email/card data.

Output JSON: `{"remember": bool, "kind": "...", "content": "..."}`.
