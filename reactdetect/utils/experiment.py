import os
import datetime

import pandas as pd

from reactdetect.utils.pandas_ops import show_df_stats
from reactdetect.utils.file_io import vim_write_zz
from reactdetect.utils.file_io import mkdir_if_dne
from reactdetect.utils.file_io import dump_json


class Experiment():

    def __init__(self):
        pass 

    def str_now(self):
        return str(datetime.datetime.now())


class PresplittedExperiment(Experiment):
    def __init__(self, train_df, test_df, name, create_subfolder=False, aux={}):
        
        assert isinstance(train_df, pd.DataFrame) and isinstance(test_df, pd.DataFrame)

        # again if same src is attacked 10 times it should all just go to either train OR test data
        s1 = set(train_df['unique_src_instance_identifier'])
        s2 = set(test_df['unique_src_instance_identifier'])
        assert len(s1.intersection(s2)) == 0

        # if two attacks happen to produce same perturbed text that's ok - they are just noisy labels, but should be really rare
        # try:
        #     assert len(set(train_df['perturbed_text']).intersection( set(test_df['perturbed_text'])) )<3,\
        #     'your train and test df have duplicate perturbed text of count : '+str(set(train_df['perturbed_text']).\
        #     intersection( set(test_df['perturbed_text'])))
        # except AssertionError:
        #     for idx in train_df.index:
        #         for tidx in test_df.index:
        #             if train_df.at[idx,'perturbed_text'] == test_df.at[tidx,'perturbed_text']:
        #                 print('*'*40)
        #                 print(train_df.at[idx,'unique_src_instance_identifier'], train_df.at[idx,'attack_name'], train_df.at[idx,'target_model'])
        #                 print('shares perturbed text with ')
        #                 print(test_df.at[tidx,'unique_src_instance_identifier'], test_df.at[tidx,'attack_name'], test_df.at[tidx,'target_model'])
        #                 print('*'*40)

        # more than one label 
        assert len(train_df['attack_name'].unique()) > 1
        assert len(test_df['attack_name'].unique()) > 1

        self.train_df, self.test_df, self.name, self.create_subfolder = train_df, test_df, name,create_subfolder 
        self.aux = aux

    def stats(self):
        out = """\n\n pre-splitted training-testing\n\n```\n"""
        out = out + 'train\n' + show_df_stats(self.train_df) + '\ntest\n' + show_df_stats(self.test_df)
        out = out + """\n```"""
        return out
    
    def to_markdown(self):
        md_info = """### """ + self.name + """\ngenerated on """ + self.str_now() + """\n### datastats\n"""
        md_info = md_info + self.stats() + """\n```"""
        return md_info

    def dump(self, exp_root_dir):
        odir = exp_root_dir
        if self.create_subfolder:
            odir = os.path.join(odir, self.name)
        print(' *** ', self.name, ' saving to ', odir)
        mkdir_if_dne(odir)
        self.train_df.to_csv(os.path.join(odir, 'train.csv'))
        self.test_df.to_csv(os.path.join(odir, 'test.csv'))
        md_info = self.to_markdown()
        vim_write_zz(os.path.join(odir, 'README.md'), md_info)

        exp_info = dict()
        
        if 'clean_vs' in self.name:
            exp_info['is_binary'] = True
        else:
            exp_info['is_binary'] = False 

        exp_info['setting'] = self.name

        exp_info.update(self.aux)

        dump_json(exp_info, os.path.join(odir, 'setting.json'))

        return self

class PresplittedExperimentNew(Experiment):
    def __init__(self, train_df, val_df, test_df_release, test_df_hidden, name, create_subfolder=False, aux={}):
        
        assert isinstance(train_df, pd.DataFrame) and isinstance(val_df, pd.DataFrame) and isinstance(test_df_release, pd.DataFrame) and isinstance(test_df_hidden, pd.DataFrame)

        # again if same src is attacked 10 times it should all just go to either train OR test data
        s1 = set(train_df['unique_src_instance_identifier'])
        s2 = set(val_df['unique_src_instance_identifier'])
        s3 = set(test_df_release['unique_src_instance_identifier'])
        s4 = set(test_df_hidden['unique_src_instance_identifier'])
        
        assert len(s1.intersection(s2)) == 0
        assert len(s1.intersection(s3)) == 0
        assert len(s1.intersection(s4)) == 0
        assert len(s2.intersection(s3)) == 0
        assert len(s2.intersection(s4)) == 0
        assert len(s4.intersection(s3)) == 0

        assert len(train_df['attack_name'].unique()) > 1
        assert len(val_df['attack_name'].unique()) > 1
        assert len(test_df_release['attack_name'].unique()) > 1
        assert len(test_df_hidden['attack_name'].unique()) > 1


        self.train_df, self.val_df, self.test_df_release, self.test_df_hidden, self.name, self.create_subfolder = train_df, val_df, test_df_release, test_df_hidden, name, create_subfolder 
        self.aux = aux

    def stats(self):
        out = """\n\n pre-splitted training-testing\n\n```\n"""
        out = out + 'train\n' + show_df_stats(self.train_df)
        out = out + '\nval\n' + show_df_stats(self.val_df)
        out = out + '\ntest release\n' + show_df_stats(self.test_df_release)
        out = out + '\ntest hidden\n' + show_df_stats(self.test_df_hidden)
        out = out + """\n```"""
        return out
    
    def to_markdown(self):
        md_info = """### """ + self.name + """\ngenerated on """ + self.str_now() + """\n### datastats\n"""
        md_info = md_info + self.stats() + """\n```"""
        return md_info

    def dump(self, exp_root_dir):
        odir = exp_root_dir
        if self.create_subfolder:
            odir = os.path.join(odir, self.name)
        print(' *** ', self.name, ' saving to ', odir)
        mkdir_if_dne(odir)
        self.train_df.to_csv(os.path.join(odir, 'train.csv'))
        self.val_df.to_csv(os.path.join(odir, 'val.csv'))
        self.test_df_release.to_csv(os.path.join(odir, 'test_release.csv'))
        self.test_df_release.to_csv(os.path.join(odir, 'test_hidden.csv'))
        md_info = self.to_markdown()
        vim_write_zz(os.path.join(odir, 'README.md'), md_info)

        exp_info = dict()
        
        if 'clean_vs' in self.name:
            exp_info['is_binary'] = True
        else:
            exp_info['is_binary'] = False 

        exp_info['setting'] = self.name

        exp_info.update(self.aux)

        dump_json(exp_info, os.path.join(odir, 'setting.json'))

        return self
