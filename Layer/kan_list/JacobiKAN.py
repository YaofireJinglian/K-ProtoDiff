import torch
import torch.nn.functional as F
import math
from torchinfo import summary

class JacobiKANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        degree=5,
        a=1.0,
        b=1.0,
        scale_base=0.5,
        scale_jacobi=0.5,
        base_activation=torch.nn.SiLU,
        use_bias=True,
    ):
        """
        初始化 JacobiKANLinear 层。

        参数:
            in_features (int): 输入特征的维度。
            out_features (int): 输出特征的维度。
            degree (int): Jacobi 多项式的最高阶数。
                该参数控制 Jacobi 多项式的阶数，决定了多项式的复杂度。
                更高的 degree 值意味着使用更高阶的多项式，可以捕捉到输入信号中的更多复杂模式。
            a (float): Jacobi 多项式的参数 a。
                控制多项式的形状，不同的参数组合可以得到不同形状的多项式。
            b (float): Jacobi 多项式的参数 b。
                同样控制多项式的形状。
            scale_base (float): 基础权重初始化的缩放因子。
                该参数用于在初始化基础权重（即 base_weight）时对初始化值进行缩放。
            scale_jacobi (float): Jacobi 系数初始化的缩放因子。
                该参数控制初始化 Jacobi 系数（jacobi_coeffs）时的值范围。
            base_activation (nn.Module): 基础激活函数类。
            use_bias (bool): 是否使用偏置项。
        """
        super(JacobiKANLinear, self).__init__()
        self.in_features = in_features  # 输入特征数
        self.out_features = out_features  # 输出特征数
        self.degree = degree  # Jacobi 多项式的最高阶数
        self.a = a  # Jacobi 多项式的参数 a
        self.b = b  # Jacobi 多项式的参数 b
        self.scale_base = scale_base  # 基础权重缩放因子
        self.scale_jacobi = scale_jacobi  # Jacobi 系数缩放因子
        self.base_activation = base_activation()  # 基础激活函数实例
        self.use_bias = use_bias  # 是否使用偏置项

        # 初始化基础权重参数，形状为 (out_features, in_features)
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        # 初始化 Jacobi 系数参数，形状为 (out_features, in_features, degree + 1)
        self.jacobi_coeffs = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, degree + 1)
        )

        if self.use_bias:
            # 初始化偏置项，形状为 (out_features,)
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        # 使用 Kaiming 初始化基础权重参数 base_weight
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)

        # 初始化 Jacobi 系数参数 jacobi_coeffs
        with torch.no_grad():
            std = self.scale_jacobi / (self.in_features * math.sqrt(self.degree + 1))
            self.jacobi_coeffs.normal_(mean=0.0, std=std)

        if self.use_bias:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.base_weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def jacobi_polynomials(self, x: torch.Tensor):
        """
        计算输入 x 的 Jacobi 多项式值。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)

        返回:
            torch.Tensor: Jacobi 多项式值，形状为 (batch_size, in_features, degree + 1)
        """
        # 将 x 缩放到 [-1, 1] 区间
        x = torch.tanh(x)

        batch_size, in_features = x.size()
        # 初始化 Jacobi 多项式张量，存储所有阶的值
        jacobi = torch.zeros(batch_size, in_features, self.degree + 1, device=x.device)
        jacobi[:, :, 0] = 1.0  # P_0(x) = 1

        if self.degree >= 1:
            jacobi[:, :, 1] = 0.5 * ((2 * (self.a + 1)) * x + (self.a - self.b))  # P_1(x)

        # 递推计算 Jacobi 多项式
        for n in range(2, self.degree + 1):
            n_minus_1 = jacobi[:, :, n - 1]  # P_{n-1}(x)
            n_minus_2 = jacobi[:, :, n - 2]  # P_{n-2}(x)

            k = n - 1
            alpha_n = 2 * k * (k + self.a + self.b) * (2 * k + self.a + self.b - 2)
            beta_n = (2 * k + self.a + self.b - 1) * (self.a ** 2 - self.b ** 2)
            gamma_n = (2 * k + self.a + self.b - 2) * (2 * k + self.a + self.b - 1) * (2 * k + self.a + self.b)
            delta_n = 2 * (k + self.a - 1) * (k + self.b - 1) * (2 * k + self.a + self.b)

            A = (beta_n + alpha_n * x) / gamma_n
            B = delta_n / gamma_n

            # 使用新张量保存当前递推结果
            next_jacobi = A * n_minus_1 - B * n_minus_2
            jacobi = torch.cat([jacobi[:, :, :n], next_jacobi.unsqueeze(2)], dim=2)

        return jacobi  # 形状为 (batch_size, in_features, degree + 1)





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
        x = x.view(-1, self.in_features)

        # 计算基础线性变换的输出
        base_output = F.linear(self.base_activation(x), self.base_weight)

        # 计算 Jacobi 多项式的值
        P_n = self.jacobi_polynomials(x)  # 形状为 (batch_size, in_features, degree + 1)

        # 计算 Jacobi 部分的输出
        # 将 jacobi_coeffs 转换为形状 (out_features, in_features, degree + 1)
        # 使用 einsum 进行高效的张量乘法
        jacobi_output = torch.einsum('bik,oik->bo', P_n, self.jacobi_coeffs)

        # 合并基础输出和 Jacobi 输出
        output = base_output + jacobi_output

        # 加上偏置项
        if self.use_bias:
            output += self.bias

        # 恢复输出张量的形状
        output = output.view(*original_shape[:-1], self.out_features)

        return output

    def regularization_loss(self, regularize_coeffs=1.0):
        """
        计算 Jacobi 系数的正则化损失。

        参数:
            regularize_coeffs (float): 正则化系数。

        返回:
            torch.Tensor: 正则化损失值。
        """
        # 计算 Jacobi 系数的 L2 范数
        coeffs_l2 = self.jacobi_coeffs.pow(2).mean()
        return regularize_coeffs * coeffs_l2

class JacobiKAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        degree=5,
        a=1.0,
        b=1.0,
        scale_base=1.0,
        scale_jacobi=1.0,
        base_activation=torch.nn.SiLU,
        use_bias=True,
    ):
        """
        初始化 JacobiKAN 模型。

        参数:
            layers_hidden (list): 每层的输入和输出特征数列表。
            degree (int): Jacobi 多项式的最高阶数。
            a (float): Jacobi 多项式的参数 a。
            b (float): Jacobi 多项式的参数 b。
            scale_base (float): 基础权重初始化时的缩放系数。
            scale_jacobi (float): Jacobi 系数初始化时的缩放系数。
            base_activation (nn.Module): 基础激活函数类。
            use_bias (bool): 是否使用偏置项。
        """
        super(JacobiKAN, self).__init__()

        # 初始化模型层
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                JacobiKANLinear(
                    in_features,
                    out_features,
                    degree=degree,
                    a=a,
                    b=b,
                    scale_base=scale_base,
                    scale_jacobi=scale_jacobi,
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