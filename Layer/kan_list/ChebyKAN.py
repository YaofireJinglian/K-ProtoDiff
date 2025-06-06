import torch
import torch.nn.functional as F
import math


class ChebyKANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        degree=10,
        scale_base=1.0,
        scale_cheby=1.0,
        base_activation=torch.nn.SiLU,
        use_bias=True,
    ):
        """
        初始化 ChebyKANLinear 层。

        参数:
            in_features (int): 输入特征的维度。
            out_features (int): 输出特征的维度。
            degree (int): Chebyshev 多项式的最高阶数。
                该参数控制 Chebyshev 多项式的阶数，决定了多项式的复杂度。
                更高的 degree 值意味着使用更高阶的多项式，可以捕捉到输入信号中的更多复杂模式。
            scale_base (float): 基础权重初始化的缩放因子。
                该参数用于在初始化基础权重（即 base_weight）时对初始化值进行缩放。
            scale_cheby (float): Chebyshev 系数初始化的缩放因子。
                该参数控制初始化 Chebyshev 系数（cheby_coeffs）时的值范围。
            base_activation (nn.Module): 基础激活函数类。
            use_bias (bool): 是否使用偏置项。
        """
        super(ChebyKANLinear, self).__init__()
        self.in_features = in_features  # 输入特征数
        self.out_features = out_features  # 输出特征数
        self.degree = degree  # Chebyshev 多项式的最高阶数
        self.scale_base = scale_base  # 基础权重缩放因子
        self.scale_cheby = scale_cheby  # Chebyshev 系数缩放因子
        self.base_activation = base_activation()  # 基础激活函数实例
        self.use_bias = use_bias  # 是否使用偏置项

        # 初始化基础权重参数，形状为 (out_features, in_features)
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        # 初始化 Chebyshev 系数参数，形状为 (out_features, in_features, degree + 1)
        self.cheby_coeffs = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, degree + 1)
        )

        if self.use_bias:
            # 初始化偏置项，形状为 (out_features,)
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        # 预计算 Chebyshev 多项式的阶数索引，形状为 (degree + 1,)
        self.register_buffer("cheby_orders", torch.arange(0, degree + 1).float())

        self.reset_parameters()

    def reset_parameters(self):
        # 使用 Kaiming 正态分布初始化基础权重参数 base_weight
        torch.nn.init.kaiming_normal_(self.base_weight, a=math.sqrt(5) * self.scale_base)

        # 使用正态分布初始化 Chebyshev 系数参数 cheby_coeffs
        with torch.no_grad():
            std = self.scale_cheby / math.sqrt(self.in_features)
            self.cheby_coeffs.normal_(0, std)

        if self.use_bias:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.base_weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.normal_(self.bias, 0, bound)

    def chebyshev_polynomials(self, x: torch.Tensor):
        """
        计算输入 x 的 Chebyshev 多项式值。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)

        返回:
            torch.Tensor: Chebyshev 多项式值，形状为 (batch_size, in_features, degree + 1)
        """
        # 将 x 缩放到 [-1, 1] 区间，并且用clamp裁剪，否则会出现nan
        x = torch.tanh(x.clamp(-1,1))
        # x = torch.tanh(x)

        # 计算 arccos(x)，以便使用 Chebyshev 多项式的三角函数定义
        theta = torch.acos(x)  # 形状为 (batch_size, in_features)

        # 计算每个阶数的 Chebyshev 多项式值
        # cheby_orders 形状为 (degree + 1,)
        # theta.unsqueeze(-1) 形状为 (batch_size, in_features, 1)
        # 计算 theta * n，形状为 (batch_size, in_features, degree + 1)
        theta_n = theta.unsqueeze(-1) * self.cheby_orders

        # 计算 cos(n * arccos(x))，得到 Chebyshev 多项式的值
        T_n = torch.cos(theta_n)  # 形状为 (batch_size, in_features, degree + 1)

        return T_n

    def forward(self, x: torch.Tensor):
        """
        实现模型的前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 (..., in_features)。

        返回:
            torch.Tensor: 输出张量，形状为 (..., out_features)。
        """
        # 保存输入张量的原始形状
        original_shape = x.shape

        # 将输入展平成二维张量，形状为 (-1, in_features)
        x = x.reshape(-1, self.in_features)

        # 计算基础线性变换的输出
        base_output = F.linear(self.base_activation(x), self.base_weight)

        # 计算 Chebyshev 多项式的值
        T_n = self.chebyshev_polynomials(x)  # 形状为 (batch_size, in_features, degree + 1)

        # 计算 Chebyshev 部分的输出
        # 将 cheby_coeffs 转换为形状 (out_features, in_features, degree + 1)
        # 使用 einsum 进行高效的张量乘法
        cheby_output = torch.einsum('bik,oik->bo', T_n, self.cheby_coeffs)

        # 合并基础输出和 Chebyshev 输出
        output = base_output + cheby_output

        # 加上偏置项
        if self.use_bias:
            output += self.bias

        # 恢复输出张量的形状
        output = output.view(*original_shape[:-1], self.out_features)
       
        return output

    def regularization_loss(self, regularize_coeffs=1.0):
        """
        计算 Chebyshev 系数的正则化损失。

        参数:
            regularize_coeffs (float): 正则化系数。

        返回:
            torch.Tensor: 正则化损失值。
        """
        # 计算 Chebyshev 系数的 L2 范数
        coeffs_l2 = self.cheby_coeffs.pow(2).mean()
        return regularize_coeffs * coeffs_l2

class ChebyKAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        degree=7,
        scale_base=0.5,
        scale_cheby=0.3,
        base_activation=torch.nn.SiLU,
        use_bias=False,
    ):
        """
        初始化 ChebyKAN 模型。

        参数:
            layers_hidden (list): 每层的输入和输出特征数列表。
            degree (int): Chebyshev 多项式的最高阶数。
            scale_base (float): 基础权重初始化时的缩放系数。
            scale_cheby (float): Chebyshev 系数初始化时的缩放系数。
            base_activation (nn.Module): 基础激活函数类。
            use_bias (bool): 是否使用偏置项。
        """
        super(ChebyKAN, self).__init__()

        # 初始化模型层
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                ChebyKANLinear(
                    in_features,
                    out_features,
                    degree=degree,
                    scale_base=scale_base,
                    scale_cheby=scale_cheby,
                    base_activation=base_activation,
                    use_bias=use_bias,
                )
            )

    def forward(self, x: torch.Tensor):
        """
        实现模型的前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 (..., in_features)。

        返回:
            torch.Tensor: 输出张量，形状为 (..., out_features)。
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def regularization_loss(self, regularize_coeffs=1.0):
        """
        计算模型的正则化损失。

        参数:
            regularize_coeffs (float): 正则化系数。

        返回:
            float: 总的正则化损失。
        """
        return sum(
            layer.regularization_loss(regularize_coeffs)
            for layer in self.layers
        )


def test_custom_tensor():
    # 1. 定义模型结构（输入维度7，输出维度32）
    layers_hidden = [7, 64, 128, 64, 7]

    # 2. 创建模型
    model = ChebyKAN(
        layers_hidden=layers_hidden,
        degree=5,
        scale_base=1.0,
        scale_cheby=1.0,
        base_activation=torch.nn.SiLU,
        use_bias=True,
    )

    # 3. 生成输入张量 [32, 96, 7]
    input_tensor = torch.randn(32, 96, 7)

    # 4. 前向传播
    output = model(input_tensor)
    print("Output shape:", output.shape)  # 应为 [32, 96, 32]


