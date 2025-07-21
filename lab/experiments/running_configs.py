class ExperimentConfig:

    def get_run_params(self):
        return self.__dict__


class DataConfigs(ExperimentConfig):
    def __init__(self,
                 **kwargs):
        for k, v in kwargs.items():
            setattr(self, k , v)


class SetupConfigs(ExperimentConfig):
    def __init__(self,
                 **kwargs):
        for k, v in kwargs.items():
            setattr(self, k , v)


if __name__ == '__main__':
    print(DataConfigs(aa=1).get_run_params())
    print(SetupConfigs().get_run_params())