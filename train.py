import torch
import torch.optim as optim
from torch.autograd import Variable

class Trainer:
    def __init__(self, args, model, criterion, logger):
        self.model = model
        self.criterion = criterion
        self.optimizer = optim.SGD(
                    model.parameters(),
                    args.learn_rate,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay)
        self.nGPU = args.nGPU
        self.learn_rate = args.learn_rate
        self.architecture = args.model

    def train(self, epoch, train_loader):
        n_batches = len(train_loader)

        color_acc_avg = 0
        type_acc_avg = 0

        loss_avg = 0
        total = 0

        model = self.model
        model.train()
        self.learning_rate(epoch)

        for i, (input_tensor, target) in enumerate(train_loader):

            color_target = target[0]
            type_target = target[1]

            if self.nGPU > 0:
                input_tensor = input_tensor.cuda()
                color_target = color_target.cuda(async=True)
                type_target = type_target.cuda(async=True)

            batch_size = color_target.size(0)
            input_var = Variable(input_tensor)
            color_target_var = Variable(color_target)
            type_target_var = Variable(type_target)

            if self.architecture == 'inception':
                color_output, type_output, color_aux, type_aux = model(input_var)

                color_out_loss = self.criterion(color_output, color_target_var)
                type_out_loss = self.criterion(type_output, type_target_var)

                color_aux_loss = self.criterion(color_aux, color_target_var)
                type_aux_loss = self.criterion(type_aux, type_target_var)

                loss = color_out_loss + type_out_loss + color_aux_loss + type_aux_loss;

            else:
                color_output, type_output = model(input_var)

                color_loss = self.criterion(color_output, color_target_var)
                type_loss = self.criterion(type_output, type_target_var)

                loss = color_loss + type_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            color_acc = self.accuracy(color_output.data, color_target)
            type_acc = self.accuracy(type_output.data, type_target)

            color_acc_avg += color_acc * batch_size
            type_acc_avg += type_acc * batch_size

            loss_avg += loss.data[0] * batch_size
            total += batch_size

            print("| Epoch[%d] [%d/%d]  Loss %1.4f  Color %6.3f  Type %6.3f" % (
                    epoch,
                    i + 1,
                    n_batches,
                    loss.data[0],
                    color_acc,
                    type_acc))

        loss_avg /= total
        color_acc_avg /= total
        type_acc_avg /= total

        print("\n=> Epoch[%d]  Loss: %1.4f  Color %6.3f  Type %6.3f\n" % (
                epoch,
                loss_avg,
                color_acc_avg,
                type_acc_avg) )

        summary = dict()

        summary['color_acc'] = color_acc_avg
        summary['type_acc'] = type_acc_avg
        summary['loss'] = loss_avg

        return summary

    def test(self, epoch, test_loader):
        n_batches = len(test_loader)

        color_acc_avg = 0
        type_acc_avg = 0

        total = 0

        model = self.model
        model.eval()

        for i, (input_tensor, target) in enumerate(test_loader):

            color_target = target[0]
            type_target = target[1]

            if self.nGPU > 0:
                input_tensor = input_tensor.cuda()
                color_target = color_target.cuda(async=True)
                type_target = type_target.cuda(async=True)

            batch_size = color_target.size(0)
            input_var = Variable(input_tensor)
            color_target_var = Variable(color_target)
            type_target_var = Variable(type_target)

            color_output, type_output = model(input_var)

            color_acc = self.accuracy(color_output.data, color_target)
            type_acc = self.accuracy(type_output.data, type_target)

            color_acc_avg += color_acc * batch_size
            type_acc_avg += type_acc * batch_size

            total += batch_size

            print("| Test[%d] [%d/%d]  Color %6.3f  Type %6.3f" % (
                    epoch,
                    i + 1,
                    n_batches,
                    color_acc,
                    type_acc))

        color_acc_avg /= total
        type_acc_avg /= total

        print("\n=> Test[%d]  Color %6.3f  Type %6.3f\n" % (
                epoch,
                color_acc_avg,
                type_acc_avg))

        summary = dict()

        summary['color_acc'] = color_acc_avg
        summary['type_acc'] = type_acc_avg

        return summary

    def accuracy(self, output, target):

        batch_size = target.size(0)

        _, pred = torch.max(output, 1)

        correct = pred.eq(target).float().sum(0)

        correct.mul_(100. / batch_size)

        return correct[0]

    def learning_rate(self, epoch):
        decay = 0.1 ** int(epoch / 10)
        learn_rate = self.learn_rate * decay
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learn_rate
