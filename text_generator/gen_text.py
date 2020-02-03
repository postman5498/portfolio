from textgenrnn import textgenrnn

textgen = textgenrnn()

# textgen.train_from_file('shakespeare_example.txt', num_epochs=1)
textgen.train_from_file("bgb.txt", num_epochs=5)

textgen.generate(20)
