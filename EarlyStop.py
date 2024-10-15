class EarlyStopping:
    def __init__(
        self, patience=5, verbose=False, delta=0, path="checkpoint.pth", trace_func=print
    ):
        """
        Args:
            patience (int): 当验证损失不再提升时，训练允许的连续下降次数
            verbose (bool): 是否打印详细信息
            delta (float): 改善的最小阈值
            path (str): 保存模型的路径
            trace_func (function): 打印函数
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0
