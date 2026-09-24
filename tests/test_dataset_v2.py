import json
from dataclasses import replace
from importlib.resources import files
import pytest
from agentprobe.dataset_v2 import expression_key, generate, check_groups
from agentprobe.tasks_io import load_train, load_test
from agentprobe.task_gen import Task
from agentprobe.prepare_v2 import prepare


def task(ops, nums):
    return Task('x','x','x',ops,nums,len(ops),0)


def test_canonical_expressions():
    assert expression_key(task(['add','add'],[1,2,3])) == expression_key(task(['add','add'],[3,1,2]))
    assert expression_key(task(['multiply','add'],[2,3,4])) == expression_key(task(['multiply','add'],[3,2,4]))
    assert expression_key(task(['add','multiply'],[1,2,3])) != expression_key(task(['multiply','add'],[1,2,3]))
    assert expression_key(task(['add'],[1,4])) != expression_key(task(['add'],[2,3]))


def test_deterministic_fresh_and_protected(tmp_path):
    a,b = tmp_path/'a',tmp_path/'b'
    generate(a,20);generate(b,20)
    for f in a.iterdir():
        assert f.read_bytes() == (b/f.name).read_bytes()
    train,test = load_train(a),load_test(a)
    check_groups(train,test)
    old=load_train(files('agentprobe').joinpath('data'))+load_test(files('agentprobe').joinpath('data'))
    check_groups(train+test,old)
    assert not any(t.family in {'f_len5_b','f_mul_add_mul'} for t in train)
    with pytest.raises(ValueError):generate(a,20)


def teacher_rows(tasks):
    for t in tasks:
        steps=[];value=t.numbers[0]
        for op,n in zip(t.reference_tools,t.numbers[1:]):
            args={'a':value,'b':n}
            value=value+n if op=='add' else value*n
            steps.append({'tool':op,'args':args,'result':value,'status':'ok','kind':'tool'})
        yield {'task_id':t.task_id,'question':t.question,'answer':t.answer,'final_answer':str(value),'steps':steps}


def test_prepare_and_corruption(tmp_path):
    data=tmp_path/'data';generate(data,20)
    teacher=tmp_path/'teacher.jsonl'
    teacher.write_text(''.join(json.dumps(e)+'\n' for e in teacher_rows(load_train(data)[:12])))
    out=tmp_path/'training';prepare(data,teacher,out)
    m=json.loads((out/'training_split.json').read_text())
    assert not set(m['train_groups']) & set(m['validation_groups'])
    assert all((out/f'{n}_conversations.jsonl').stat().st_size for n in ['train','validation'])
    with pytest.raises(ValueError):prepare(data,teacher,out)
    (data/'test_tasks.jsonl').write_text('corrupted')
    with pytest.raises(ValueError,match='hash'):prepare(data,teacher,tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_group_overlap_detected():
    a=task(['add'],[2,3]);b=replace(a,numbers=[3,2])
    with pytest.raises(ValueError,match='overlap'):check_groups([a],[b])
