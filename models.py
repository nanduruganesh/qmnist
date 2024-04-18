import torchquantum as tq
from torchquantum.plugin import (
    tq2qiskit_measurement,
    qiskit_assemble_circs,
    op_history2qiskit,
    op_history2qiskit_expand_params,
)
import torch
import torch.nn as nn
import torch.nn.functional as F

# 2 bit classification
class QFCModel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(
                n_ops=50, wires=list(range(self.n_wires))
            )

            # gates with trainable parameters
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)

            # some trainable gates (instantiated ahead of time)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])

            # add some more non-parameterized gates (add on-the-fly)
            qdev.h(wires=3)  # type: ignore
            qdev.sx(wires=2)  # type: ignore
            qdev.cnot(wires=[3, 0])  # type: ignore
            qdev.rx(
                wires=1,
                params=torch.tensor([0.1]),
                static=self.static_mode,
                parent_graph=self.graph,
            )  # type: ignore

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_u3_h_rx"])

        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=x.shape[0], device=x.device, record_op=True
        )

        bsz = x.shape[0]
        print(x.shape)
        x = F.avg_pool2d(x, 6).view(bsz, 16)
        print(x.shape)
        devi = x.device

        if use_qiskit:
            # use qiskit to process the circuit
            # create the qiskit circuit for encoder
            self.encoder(qdev, x)  
            op_history_parameterized = qdev.op_history
            qdev.reset_op_history()
            encoder_circs = op_history2qiskit_expand_params(self.n_wires, op_history_parameterized, bsz=bsz)

            # create the qiskit circuit for trainable quantum layers
            self.q_layer(qdev)
            op_history_fixed = qdev.op_history
            qdev.reset_op_history()
            q_layer_circ = op_history2qiskit(self.n_wires, op_history_fixed)

            # create the qiskit circuit for measurement
            measurement_circ = tq2qiskit_measurement(qdev, self.measure)

            # assemble the encoder, trainable quantum layers, and measurement circuits
            assembled_circs = qiskit_assemble_circs(
                encoder_circs, q_layer_circ, measurement_circ
            )

            # call the qiskit processor to process the circuit
            x0 = self.qiskit_processor.process_ready_circs(qdev, assembled_circs).to(  # type: ignore
                devi
            )
            x = x0

        else:
            # use torchquantum to process the circuit
            self.encoder(qdev, x)
            qdev.reset_op_history()
            self.q_layer(qdev)
            x = self.measure(qdev)

        # x = x.reshape(bsz, 2, 2).sum(-1).squeeze()
        x = x.reshape(bsz, 4)
            
        x = F.log_softmax(x, dim=1)

        return x

# 4 bit classification
class EightQNN(tq.QuantumModule):
    class EightQLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 8
            self.random_layer = tq.RandomLayer(
                n_ops=50, wires=list(range(self.n_wires))
            )

            # gates with trainable parameters
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

            self.rx1 = tq.RX(has_params=True, trainable=True)
            self.ry1 = tq.RY(has_params=True, trainable=True)
            self.rz1 = tq.RZ(has_params=True, trainable=True)
            self.crx1 = tq.CRX(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)

            # some trainable gates (instantiated ahead of time)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])

            self.rx1(qdev, wires=4)
            self.ry1(qdev, wires=5)
            self.rz1(qdev, wires=7)
            self.crx1(qdev, wires=[4, 6])

            # add some more non-parameterized gates (add on-the-fly)
            qdev.h(wires=3)  # type: ignore
            qdev.sx(wires=2)  # type: ignore
            qdev.cnot(wires=[3, 0])  # type: ignore
            qdev.rx(
                wires=1,
                params=torch.tensor([0.1]),
                static=self.static_mode,
                parent_graph=self.graph,
            )  # type: ignore

            qdev.h(wires=7)  # type: ignore
            qdev.sx(wires=6)  # type: ignore
            qdev.cnot(wires=[7, 4])  # type: ignore
            qdev.rx(
                wires=5,
                params=torch.tensor([0.1]),
                static=self.static_mode,
                parent_graph=self.graph,
            )  # type: ignore

    def __init__(self):
        super().__init__()
        self.n_wires = 8
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["8x2_ry"])

        self.q_layer = self.EightQLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=x.shape[0], device=x.device, record_op=True
        )

        #x.shape is equal to [256,1,28,28]
        bsz = x.shape[0]
        x = F.avg_pool2d(x, kernel_size=5, stride=3).view(bsz, 64)
        devi = x.device

        if use_qiskit:
            # use qiskit to process the circuit
            # create the qiskit circuit for encoder
            self.encoder(qdev, x)  
            op_history_parameterized = qdev.op_history
            qdev.reset_op_history()
            encoder_circs = op_history2qiskit_expand_params(self.n_wires, op_history_parameterized, bsz=bsz)

            # create the qiskit circuit for trainable quantum layers
            self.q_layer(qdev)
            op_history_fixed = qdev.op_history
            qdev.reset_op_history()
            q_layer_circ = op_history2qiskit(self.n_wires, op_history_fixed)

            # create the qiskit circuit for measurement
            measurement_circ = tq2qiskit_measurement(qdev, self.measure)

            # assemble the encoder, trainable quantum layers, and measurement circuits
            assembled_circs = qiskit_assemble_circs(
                encoder_circs, q_layer_circ, measurement_circ
            )

            # call the qiskit processor to process the circuit
            x0 = self.qiskit_processor.process_ready_circs(qdev, assembled_circs).to(  # type: ignore
                devi
            )
            x = x0

        else:
            # use torchquantum to process the circuit
            self.encoder(qdev, x)
            qdev.reset_op_history()
            self.q_layer(qdev)
            x = self.measure(qdev)

        x = x.reshape(bsz, 4, 2).sum(-1).squeeze()
        # x = x.reshape(bsz, 4)
            
        x = F.log_softmax(x, dim=1)

        return x

class TwentyQNN(tq.QuantumModule):
    class TwentyQLayer(tq.QuantumModule):
        def __init__(self, n_wires):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(
                n_ops=50, wires=list(range(self.n_wires))
            )

            # gates with trainable parameters
            for i in range(self.n_wires):
                exec(f"self.rx{i} = tq.RX(has_params=True, trainable=True)")

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)

            # some trainable gates (instantiated ahead of time)
            for i in range(self.n_wires):
                exec(f"self.rx{i}(qdev,wires={i})")

    def __init__(self):
        super().__init__()
        self.n_wires = 10
        
        self.encoder = tq.GeneralEncoder( 
            [   {'input_idx': [i], 'func': 'rx', 'wires': [i]} for i in range(self.n_wires) ]
        )

        self.q_layer = self.TwentyQLayer(self.n_wires)
        
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=x.shape[0], device=x.device, record_op=True
        )

        #x.shape is equal to [256,1,28,28]
        bsz = x.shape[0]
        x = F.avg_pool2d(x, kernel_size=9, stride=2)
        x = x.view(bsz, -1)
        devi = x.device

        if use_qiskit:
            # use qiskit to process the circuit
            # create the qiskit circuit for encoder
            self.encoder(qdev, x)  
            op_history_parameterized = qdev.op_history
            qdev.reset_op_history()
            encoder_circs = op_history2qiskit_expand_params(self.n_wires, op_history_parameterized, bsz=bsz)

            # create the qiskit circuit for trainable quantum layers
            self.q_layer(qdev)
            op_history_fixed = qdev.op_history
            qdev.reset_op_history()
            q_layer_circ = op_history2qiskit(self.n_wires, op_history_fixed)

            # create the qiskit circuit for measurement
            measurement_circ = tq2qiskit_measurement(qdev, self.measure)

            # assemble the encoder, trainable quantum layers, and measurement circuits
            assembled_circs = qiskit_assemble_circs(
                encoder_circs, q_layer_circ, measurement_circ
            )

            # call the qiskit processor to process the circuit
            x0 = self.qiskit_processor.process_ready_circs(qdev, assembled_circs).to(  # type: ignore
                devi
            )
            x = x0

        else:
            # use torchquantum to process the circuit
            self.encoder(qdev, x)
            qdev.reset_op_history()
            self.q_layer(qdev)
            x = self.measure(qdev)

       #x = x.sum(-1).squeeze()
        x = x.reshape(bsz, 10)
            
        x = F.log_softmax(x, dim=1)

        return x

class LayeredQNN(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(
                n_ops=50, wires=list(range(self.n_wires))
            )

            # gates with trainable parameters
            for i in range(self.n_wires):
                exec(f"self.rx{i} = tq.RX(has_params=True, trainable=True)")

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)

            # some trainable gates (instantiated ahead of time)
            for i in range(self.n_wires):
                exec(f"self.rx{i}(qdev,wires={i})")

    def __init__(self, n_wires, q_layers):
        super().__init__()
        self.n_wires = n_wires
        self.q_layers = q_layers
        
        self.encoder = tq.GeneralEncoder( 
            [   {'input_idx': [i], 'func': 'rx', 'wires': [i]} for i in range(self.n_wires) ]
        )

        #self.q_layer = self.QLayer(self.n_wires)
        for i in range(self.q_layers):
            exec(f"self.q_layer{i} = self.QLayer(self.n_wires)")
        
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=x.shape[0], device=x.device, record_op=True
        )

        #x.shape is equal to [256,1,28,28]
        bsz = x.shape[0]
        #x = F.avg_pool2d(x, kernel_size=9, stride=2)
        x = x.view(bsz, -1)
        devi = x.device

        self.encoder(qdev, x)
        qdev.reset_op_history()
        #self.q_layer(qdev)
        for i in range(3):
            exec(f"self.q_layer{i}(qdev)")

        x = self.measure(qdev)

       #x = x.sum(-1).squeeze()
        x = x[:, :10]
        #x = x.reshape(bsz, 10)
            
        x = F.log_softmax(x, dim=1)

        return x


class HybridQNN(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires, func):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(
                n_ops=50, wires=list(range(self.n_wires))
            )
            self.func = func
            # gates with trainable parameters
            # Example evaluation of this line: self.rx4 = tq.RX(has_params=True, trainable=True)
            for i in range(self.n_wires):
                exec(f"self.{self.func}{i} = tq.{func.upper()}(has_params=True, trainable=True)")

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)

            # some trainable gates (instantiated ahead of time)
            for i in range(self.n_wires):
                # Example evaluation of this line: self.rx4(qdev, wires=4)
                exec(f"self.{self.func}{i}(qdev,wires={i})")

    def __init__(self, n_wires, q_layers, func='rx'):
        super().__init__()
        self.n_wires = n_wires
        self.q_layers = q_layers
        self.func = func
        
        self.encoder = tq.GeneralEncoder( 
           [   {'input_idx': [i], 'func': self.func, 'wires': [i]} for i in range(self.n_wires) ]
        )

        for i in range(self.q_layers):
            exec(f"self.q_layer{i} = self.QLayer(self.n_wires, self.func)")

        self.linear = nn.Linear(16, 50)
        self.act = nn.ReLU()

        self.linear2 = nn.Linear(50, 10)
        self.act2 = nn.ReLU()

        self.linear3 = nn.Linear(10, 10)
        #self.act3 = nn.ReLu()
        
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=x.shape[0], device=x.device, record_op=True
        )

        #x.shape is equal to [256,1,28,28]
        bsz = x.shape[0]
        x = F.avg_pool2d(x, kernel_size=6, stride=6)
        x = x.reshape(bsz, -1)

        x = self.linear(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.act2(x)

        devi = x.device

        self.encoder(qdev, x)
        qdev.reset_op_history()
        
        for i in range(self.q_layers):
            exec(f"self.q_layer{i}(qdev)")

        x = self.measure(qdev)

       #x = x.sum(-1).squeeze()
        #x = x[:, :10]
        #x = x.reshape(bsz, 10)
        x = self.linear3(x)
        x = F.log_softmax(x, dim=1)

        return x
