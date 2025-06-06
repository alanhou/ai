from langchain_openai.chat_models import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")

completion = model.invoke("Hi there!")
print(completion.content)
# Hello! How can I assist you today?

completions = model.batch(["Hi there!", "Bye!"])
print([c.content for c in completions])
# ['Hello! How can I assist you today?', 'Goodbye! If you have more questions in the future, feel free to ask. Take care!']

for token in model.stream("Bye!"):
    print(token.content)
    # Good
    # bye
    # !
    #  If
    #  you
    #  have
    #  any
    #  more
    #  questions
    #  in
    #  the
    #  future
    # ,
    #  feel
    #  free
    #  to
    #  ask
    # .
    #  Take
    #  care
    # !