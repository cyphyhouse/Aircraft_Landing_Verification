import torch
import torch.nn.functional as F

mult = None

def get_model(num_dim_input, num_dim_output, config, args):
    global mult
    model = torch.nn.Sequential(
            torch.nn.Linear(num_dim_input, args.layer1, bias=False),
            torch.nn.Tanh(),
            torch.nn.Linear(args.layer1, args.layer2, bias=False),
            torch.nn.Tanh(),
            torch.nn.Linear(args.layer2, num_dim_output*num_dim_output, bias=False))

    if hasattr(config, 'get_xt_scale'):
        scale = config.get_xt_scale()
        mult = torch.diag(torch.from_numpy(scale))
    else:
        mult = None

    def forward(input):
        global mult
        output = model(input)
        output = output.view(input.shape[0], num_dim_output, num_dim_output)
        if mult is not None:
            mult = mult.type(input.type())
            output = torch.matmul(output, mult)
        return output
    return model, forward

def get_model_rect(num_dim_input, num_dim_output, layer1, layer2):
    global mult
    model = torch.nn.Sequential(
            # torch.nn.BatchNorm1d(num_dim_input),
            torch.nn.Linear(num_dim_input, layer1),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(layer1, layer2),
            torch.nn.ReLU(),
            # torch.nn.BatchNorm1d(layer2),
            torch.nn.Linear(layer2, num_dim_output))

    mult = None

    def forward(input):
        global mult
        output = model(input)
        output = output.view(input.shape[0], num_dim_output)
        if mult is not None:
            mult = mult.type(input.type())
            output = torch.matmul(output, mult)
        return output
    return model, forward

def get_model_rect4(num_dim_input, num_dim_output, layer1, layer2, layer3):
    global mult
    model = torch.nn.Sequential(
            torch.nn.Linear(num_dim_input, layer1),
            # torch.nn.BatchNorm1d(layer1),
            torch.nn.PReLU(),
            torch.nn.Linear(layer1, layer2),
            # torch.nn.BatchNorm1d(layer2),
            # torch.nn.Sigmoid(),
            torch.nn.ReLU(),
            torch.nn.Linear(layer2, layer3),
            # torch.nn.BatchNorm1d(layer2),
            torch.nn.ReLU(),
            torch.nn.Linear(layer3, num_dim_output))

    mult = None

    def forward(input):
        global mult
        output = model(input)
        output = output.view(input.shape[0], num_dim_output)
        if mult is not None:
            mult = mult.type(input.type())
            output = torch.matmul(output, mult)
        return output
    return model, forward

def get_model_rect2(num_dim_input, num_dim_output, layer1, layer2, layer3):
    global mult
    model = torch.nn.Sequential(
            torch.nn.Linear(num_dim_input, layer1),
            torch.nn.BatchNorm1d(layer1),
            torch.nn.Sigmoid(),
            torch.nn.Linear(layer1, layer2),
            torch.nn.BatchNorm1d(layer2),
            torch.nn.Sigmoid(),
            torch.nn.Linear(layer2, layer3),
            # torch.nn.BatchNorm1d(layer2),
            torch.nn.ReLU(),
            torch.nn.Linear(layer3, num_dim_output))

    mult = None

    def forward(input):
        global mult
        output = model(input)
        output = output.view(input.shape[0], num_dim_output)
        if mult is not None:
            mult = mult.type(input.type())
            output = torch.matmul(output, mult)
        return output
    return model, forward

def get_model_rect3(num_dim_input, num_dim_output, layer1):
    global mult
    model = torch.nn.Sequential(
            torch.nn.Linear(num_dim_input, layer1),
            torch.nn.ReLU(),
            torch.nn.Linear(layer1, num_dim_output),
    )
    
    mult = None

    def forward(input):
        global mult
        output = model(input)
        output = output.view(input.shape[0], num_dim_output)
        if mult is not None:
            mult = mult.type(input.type())
            output = torch.matmul(output, mult)
        return output
    return model, forward