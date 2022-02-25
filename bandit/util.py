import json
import os


def arm_components(arms):
    """
    Get arm weights as list of dicts.
    :param arms: List[Arm]
    :return:
        List[Dict]
    """
    return [
        {'name': arm.name,
         'weights': {k: v for k, v in arm.__dict__.items() if k != 'name'}
         }
        for arm in arms
    ]


def experiment_params(experiment):
    return experiment.__dict__


def agent_params(agent):
    return {'name': agent.__class__.name,
            'params': {k: v for k, v in agent.__dict__.items()
                       if k not in ['callbacks', 'arms', 'experiment']
                       and not k.startswith('_')
                       }
            }


def agent_component_parts(agent):
    """
    Divides to agent / process into three
    functional parts:
        arms, experiment, parameters
    :param agent: process.Process / Agent
    :return:
        Tuple(List[Arm], Experiment, Dict)
    """
    return (
        arm_components(agent.__dict__.get('arms', [])),
        experiment_params(agent.__dict__.get('experiment', None)),
        agent_params(agent)
    )


def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)


def read_json(path):
    file = open(path, 'r')
    data = json.load(file)
    file.close()
    return data


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
