import fasttext

# 禁用 mmap
model = fasttext.load_model(r"/mnt/d/code/项目/cs336/CS336-Chinese-co-construction/coursework/Assignment4_Data/cs336_data/lid.176.bin", mmap=False)
print(model.predict("This is a test."))