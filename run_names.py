from ikeanames.ikeanames import *

class Runner:
    def __init__(self):
        self.names = load_names()
        (self.max_len, self.encoding, self.decoding) = get_encoding(self.names)

    def train(self):
        (X, y) = make_train_set(self.names, self.encoding, self.decoding, self.max_len)
        model = make_model(X)
        train_model(model, X, y, epochs=1000)
        try:
            fn = sys.argv[2]
        except IndexError:
            fn = None
        save_model(model, fn)

    def predict(self, n = 10):
        model = load_model('models/model.h5')
        names = []
        for i in range(n):
            name = predict_name(model, self.max_len, self.encoding, self.decoding)
            if name in self.names:
                continue
            names.append(name)
        return names

if __name__ == '__main__':
    r = Runner()
    #r.train()
    names = r.predict(300)
    with open('result.txt','w') as stream:
        for name in names:
            stream.write(name+"\n")
