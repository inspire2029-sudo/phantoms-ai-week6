# W6D1 — Meet the LLM

## 1. What is a token?

A token is a small piece of text that the language model processes. A token is not always the same as one complete word: a word can be one token, several tokens, or sometimes part of a word can be a token.

Tokens matter because the model does not receive text as an unlimited stream of characters. The input and output are processed within a context window, and token usage can affect limits, speed, and API cost.

## 2. Why is the model described as predicting instead of understanding?

A language model generates text by repeatedly predicting what token is likely to come next based on the tokens that came before it and the patterns learned during training.

Calling this prediction does not mean the model is useless or that its output is random. The predictions can encode a large amount of useful structure and knowledge. However, next-token prediction is different from having human-style understanding, intentions, experiences, or guaranteed factual reasoning.

## 3. Roles: system, user, assistant

The conversation is structured into roles.

- **system**: defines high-level behavior and instructions.
- **user**: contains the user's request.
- **assistant**: contains the model's generated response.

In this task, the same question is sent twice while changing only the system instruction. This demonstrates that the system role can strongly influence the style and level of detail of the response.

## 4. Does the API remember previous requests?

A basic API request should be treated as stateless. The application is responsible for sending the relevant conversation history again when previous context is needed.

That means a chat application can appear to have memory, while the application itself is actually managing and resending the conversation context.

## 5. Example of an incorrect or weak model answer

A useful example is asking a model a question that requires exact counting, such as:

> How many letters are in the word "mississippi"?

A model can sometimes give an incorrect count even though the word is simple. The reason is that the model is generating an answer from learned patterns rather than using a guaranteed symbolic counting procedure every time. Tokenization also means the model does not necessarily process each letter as a separate unit.

This is a good reminder that fluent language is not the same thing as guaranteed correctness.

## 6. What this practical task demonstrates

This exercise connects the concepts to a real API call:

1. The API key is stored outside the source code.
2. Python sends a system message and a user message.
3. The model returns an assistant message.
4. The same question is tested with two different system prompts.
5. Token usage is read from the API response instead of being guessed.

The API key must never be committed to GitHub.
