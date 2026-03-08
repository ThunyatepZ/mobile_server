from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)
memory.save_context({"input": "Hello"}, {"output": "Hi"})
print(memory.load_memory_variables({}))
