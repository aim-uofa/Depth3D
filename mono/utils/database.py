from types import ModuleType
import data_info

def load_data_info(module_name, data_info={}, module=None):
    if module is None:
        module = globals().get(module_name, None)
    if module:
        for key, value in module.__dict__.items():
            if not (key.startswith('__')) and not (key.startswith('_')):
                if key == 'data_info':
                    data_info.update(value)
                elif isinstance(value, ModuleType):
                    load_data_info(module_name + '.' + key, data_info, module=value)
    else:
        raise RuntimeError(f'Try to access "data_info", but cannot find {module_name} module.')

def reset_ckpt_path(cfg, data_info):
    if isinstance(cfg, dict):
        for key in cfg.keys():
            if key == 'backbone':
                new_ckpt_path = data_info['checkpoint']['database_root'] + '/' + data_info['checkpoint'][cfg.backbone.type]
                cfg.backbone.update(checkpoint=new_ckpt_path)
                continue
            elif isinstance(cfg.get(key), dict):
                reset_ckpt_path(cfg.get(key), data_info)
            else:
                continue
    else:
        return

if __name__ == '__main__':
    data_info_tmp = {}
    load_data_info('data_info', data_info_tmp)
    print('results', data_info_tmp.keys())