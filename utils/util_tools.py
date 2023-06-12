"""保存模型"""
import os

import torch


def write_record(path, message):
    file_obj = open(path, 'a')
    file_obj.write('{}\n'.format(message))
    file_obj.close()


def save_model(self, i, train_loss, valid_loss, test_loss, train_metric, val_metric, test_metric, val_metric_list,
               test_metric_list, val_importantValue):
    val_metric_list.append(val_metric)
    test_metric_list.append(test_metric)
    path = os.path.join(self.log_dir, 'saved_model')
    os.makedirs(path, exist_ok=True)
    if self.config['save_model'] == 'best_valid':
        if (self.config['task'] == 'regression' and (self.best_metric > val_metric)) or (
                self.config['task'] == 'classification' and self.change_status):
            self.best_metric = val_importantValue['average'][4]
            self.change_status = False
            torch.save(self.net.state_dict(),
                       os.path.join(self.log_dir + '/saved_model/', 'best_valid_model.pth'))
    self.writer.add_scalar('valid_loss', valid_loss, global_step=i)
    self.writer.add_scalar('test_loss', test_loss, global_step=i)

    if self.config['task'] == 'classification':
        print(f'train_loss:{train_loss} val_loss:{valid_loss} test_loss:{test_loss}\n'
              f'train_F1Measure:{train_metric} val_F1Measure:{val_metric} test_F1Measure:{test_metric}')
    else:
        pass
    write_record(self.txtfile,
                 f'epoch:{i} train_loss:{train_loss} val_loss:{valid_loss} test_loss:{test_loss}\n'
                 f'train_metric:{train_metric} val_metric:{val_metric} test_metric:{test_metric}')
    if i % self.config['save_ckpt'] == 0:
        self.save_ckpt(i)
    print("=" * 180 + '\n\n')
    return val_metric_list, test_metric_list
