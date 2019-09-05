class Model:
    # Initialize the model.
    #
    # dist: {sample: () -> np.array([dim])}
    # model: {predict: np.array([n_pts,dim]) -> {0, 1}}
    def __init__(self, dist, model):
        self.dist = dist
        self.model = model

    # Sample a random prediction.
    #
    # return: {0, 1} (a random prediction)
    def sample(self):
        x = self.dist.sample()[0]
        y = self.model.predict(x.reshape([1,-1]))
        return y
