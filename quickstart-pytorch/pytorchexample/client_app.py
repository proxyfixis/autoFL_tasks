"""pytorchexample: A Flower / PyTorch app."""

import torch
import wandb
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import Net, load_local_image_data

from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn

from pytorchexample.task import class_wise_accuracy


# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    batch_size = context.run_config["batch-size"]
    client_id = context.node_config["partition-id"]
    trainloader, _ = load_local_image_data(
        client_id=client_id,
        batch_size=batch_size
    )

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    wandb.init(
    project="TASK4-AutoFL",
    name=f"client-{context.node_config['partition-id']}",
    reinit=True,
)
    """Evaluate the model on local data."""
    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    batch_size = context.run_config["batch-size"]
    client_id = context.node_config["partition-id"]
    _, valloader = load_local_image_data(
        client_id=client_id,
        batch_size=batch_size
    )


    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    class_acc = class_wise_accuracy(model, valloader, device)

    wandb.log(
    {
        "round": context.run_config.get("current-round", 0),
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        **class_acc,
    }
)

    # Construct and return reply Message
    metrics = {
    "eval_loss": eval_loss,
    "eval_acc": eval_acc,
    "num-examples": len(valloader.dataset),
    **class_acc,
}
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    wandb.finish()
    return Message(content=content, reply_to=msg)
