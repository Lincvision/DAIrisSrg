import os

class Create_file(object):
    def __init__(self, config):
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.result_path_next = os.path.join(config.result_path, config.model_type)
        self.inner_path = config.inner_path
        self.output_path = config.output_path
        self.pseudo_file = os.path.join(config.pseudo_file, '%s-%s' % (config.source_dataset, config.target_dataset))

    def build_file(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.pseudo_file):
            os.makedirs(self.pseudo_file)
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        if not os.path.exists(self.result_path_next):
            os.makedirs(self.result_path_next)
        if not os.path.exists(self.inner_path):
            os.makedirs(self.inner_path)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)


class Create_file_2(object):
    def __init__(self, paht_1,paht_2):
        self.path1 = paht_1
        self.path2 = paht_2

    def build_file(self):
        if not os.path.exists(self.path1):
            os.makedirs(self.path1)
        if not os.path.exists(self.path2):
            os.makedirs(self.path2)
