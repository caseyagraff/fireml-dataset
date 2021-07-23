import pickle
import os


class PickleSaveLoadMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # type: ignore

    def save_name(self) -> str:
        raise NotImplementedError()

    def save_data(self) -> dict:
        raise NotImplementedError()

    def save(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(os.path.join(save_dir, self.save_name()), "wb") as f_out:
            pickle.dump(self.save_data(), f_out, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, file_name):
        with open(file_name, "rb") as f_in:
            data = pickle.load(f_in)  # nosec

        return cls(**data)
