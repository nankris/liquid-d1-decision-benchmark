"""Shared decision criteria and chat prompts; labels are canonical strings."""
from dataclasses import dataclass

CATEGORIES = {
    'ACCOUNT': 'creating, deleting, editing or switching an account, password recovery, registration problems',
    'CANCEL': 'questions about cancellation or early-termination fees/charges',
    'CONTACT': 'wants to reach customer service or talk to a human agent',
    'DELIVERY': 'delivery options and how long delivery takes',
    'FEEDBACK': 'complaints, formal claims, or leaving a review',
    'INVOICE': 'viewing, finding or getting a copy of an invoice or bill',
    'SUBSCRIPTION': 'subscribing to or unsubscribing from the newsletter',
    'ORDER': 'placing, changing, cancelling or tracking an order',
    'PAYMENT': 'accepted payment methods or problems making a payment',
    'REFUND': 'refund policy, requesting a refund, or checking refund status',
    'SHIPPING': 'setting up or changing the shipping address',
}
STARS = ['1 star: very negative', '2 stars: mostly negative', '3 stars: mixed or average',
         '4 stars: mostly positive', '5 stars: very positive']
SYSTEM = 'You are a careful classifier. Reply with only the answer token, nothing else.'

@dataclass(frozen=True)
class Task:
    name: str
    kind: str
    instructions: str
    criteria: object = None

    @property
    def options(self):
        return ['0', '1'] if self.kind == 'noul' else list(self.criteria) if self.kind == 'choice' else [str(i) for i in range(len(self.criteria))]

    @property
    def tokens(self):
        if self.kind == 'noul':
            return {'no': '0', 'yes': '1'}
        if self.kind == 'choice':
            return {chr(65 + i): label for i, label in enumerate(self.options)}
        return {str(i + 1): str(i) for i in range(len(self.criteria))}

    def question(self):
        q = {'type': self.kind, 'instructions': self.instructions}
        if self.criteria is not None:
            q['criteria'] = self.criteria
        return q

    def messages(self, text):
        if self.kind == 'noul':
            options = 'yes) yes\nno) no'
        elif self.kind == 'choice':
            options = '\n'.join(f'{token}) {label} — {self.criteria[label]}' for token, label in self.tokens.items())
        else:
            options = '\n'.join(f'{i + 1}) {description}' for i, description in enumerate(self.criteria))
        return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': f'{self.instructions}\n{options}\n\nText:\n{text}\n\nAnswer:'}]

TASKS = {
    'spam': Task('spam', 'noul', 'Is this SMS message spam (unsolicited advertising, scams, prize or premium-rate number promotions)?'),
    'toxic': Task('toxic', 'noul', 'Is this tweet offensive (insults, profanity, threats, or targeted abuse)?'),
    'routing': Task('routing', 'choice', 'Which support team should handle this customer message?', CATEGORIES),
    'rating': Task('rating', 'score', 'How many stars did this reviewer give?', STARS),
}
