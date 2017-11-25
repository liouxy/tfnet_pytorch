def linear(pic):
    return pic*255.0

class Stretch:
    def __call__(self, pic):
        return linear(pic)
