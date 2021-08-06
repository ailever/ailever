class Scaler:
    def standard(self, X, inverse=False, return_statistics=False):
        if not inverse:
            self.mean = X.mean(dim=0, keepdim=True)
            self.std = X.std(dim=0, keepdim=True)
            X = (X - self.mean)/ self.std
            if return_statistics:
                return X, (self.mean, self.std)
            else:
                return X
        else:
            X = X * self.std + self.mean
            if return_statistics:
                return X, (self.mean, self.std)
            else:
                return X

    def minmax(self, X, inverse=False, return_statistics=False):
        if not inverse:
            self.min = X.min(dim=0, keepdim=True).values
            self.max = X.max(dim=0, keepdim=True).values
            X = (X - self.min)/ (self.max - self.min)
            if return_statistics:
                return X, (self.max, self.min)
            else:
                return X
        else:
            X = X*(self.max - self.min) + self.min
            if return_statistics:
                return X, (self.min, self.max)
            else:
                return X

    def fft(self, X, inverse=False):
        if not inverse:
            X = torch.fft.fft(X)
            return X
        else:
            X = torch.fft.ifft(X)
            return X

