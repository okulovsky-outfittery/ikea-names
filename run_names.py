from ikeanames.ikeanames import predict_name, load_model, get_encoding, load_names

if __name__ == '__main__':
    model = load_model()
    names = load_names()
    (max_len, encoding, decoding) = get_encoding(load_names()[:1])
    name = predict_name(model, max_len, encoding, decoding)
    print(name)