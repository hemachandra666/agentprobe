"""CPU orchestration check; real TRL checkpoint restore needs a GPU smoke test."""
import hashlib
import importlib
import json
import sys
from types import SimpleNamespace, ModuleType


def test_selected_trainer_saved_and_stale_inputs_rejected(tmp_path, monkeypatch):
    data=tmp_path/'data';data.mkdir();out=tmp_path/'model'
    m={'schema_version':'2.0'}
    for n in ('train','validation'):
        content=b'{}\n';(data/f'{n}_conversations.jsonl').write_bytes(content)
        m[n+'_sha256']=hashlib.sha256(content).hexdigest()
    (data/'training_split.json').write_text(json.dumps(m))
    config={};events=[]
    class Model:
        pass
    class Tokenizer:
        def save_pretrained(self,p):events.append('tokenizer')
    class Fast:
        @staticmethod
        def from_pretrained(**kw):return Model(),Tokenizer()
        @staticmethod
        def get_peft_model(model,**kw):return model
    class Dataset:
        def map(self,f):return self
    class Trainer:
        def __init__(self,**kw):
            self.state=SimpleNamespace(best_model_checkpoint=None,best_metric=None,global_step=0)
        def train(self):
            self.state=SimpleNamespace(best_model_checkpoint='checkpoint-2',best_metric=.1,global_step=4)
            events.append('restored_best')
        def save_model(self,p):
            assert events==['restored_best'];out.mkdir();events.append('saved_selected')
        def save_state(self):events.append('state')
    for name,attrs in [('unsloth',{'FastLanguageModel':Fast}),('datasets',{'load_dataset':lambda *a,**k:Dataset()}),('trl',{'SFTTrainer':Trainer,'SFTConfig':lambda **k:config.update(k)})]:
        mod=ModuleType(name);mod.__dict__.update(attrs);monkeypatch.setitem(sys.modules,name,mod)
    monkeypatch.setattr(sys,'argv',['train','--data-dir',str(data),'--output-dir',str(out),'--select-best'])
    sys.modules.pop('agentprobe.train_student',None)
    module=importlib.import_module('agentprobe.train_student')
    try:
        module.main()
        assert config['save_strategy']==config['eval_strategy']=='epoch'
        assert config['load_best_model_at_end'] and not config['greater_is_better']
        assert config['metric_for_best_model']=='eval_loss'
        assert json.loads((out/'selection.json').read_text())['best_model_checkpoint']=='checkpoint-2'
        import pytest
        with pytest.raises(ValueError,match='new output'):module.main()
        monkeypatch.setattr(sys,'argv',['train','--data-dir',str(data),'--output-dir',str(tmp_path/'other'),'--select-best'])
        (data/'train_conversations.jsonl').write_text('changed')
        with pytest.raises(ValueError,match='changed'):module.main()
    finally:
        sys.modules.pop('agentprobe.train_student',None)
