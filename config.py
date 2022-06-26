
class _Config:
    def __init__(self):
        self.device=1
        self.in_len=144
        self.out_len=12
        
        self.granularity='day' # 'hour' or 'day'
        self.year='2014' # 'all' or '2015' or '2014'
        self.model='transformer' # 'convlstm' or 'transformer'

        self.lr=8e-4
        self.gamma=0.7
        self.batch_size=32
        self.epoch_num=100
        
        self.test_iteration_interval=100

        self.height=43
        self.width=56
        self.output_channels=1

        self.add_traffic=False

        # settings for convlstm
        self.middle_channels1=32
        self.middle_channels2=96
        # settings for transformer
        self.layers=3
        self.embed_size=60
        self.heads=3

        self.dynamic_settings()
    
    def dynamic_settings(self):
        if self.granularity=='day':
            self.in_len=14
            self.out_len=2
        self.max_iterations=self.epoch_num*self.test_iteration_interval
        self.lr_step_size=20*self.test_iteration_interval
        if self.year=='all':
            self.add_traffic=False # No all traffic data
        if self.add_traffic:
            self.input_channels=4  if self.year=='2014' else 3
        else:
            self.input_channels=1
        self.exp_name='{}-{}-{}'.format(self.year, self.granularity, self.model)
        if self.model=='transformer':
            self.exp_name+='-layer{}'.format(self.layers)
        if not self.add_traffic:
            self.exp_name+='-no-traffic'

    


global_config = _Config()