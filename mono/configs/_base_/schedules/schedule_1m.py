optimizer = dict(
    type='SGD', 
    encoder=dict(lr=0.01, ),
    decoder=dict(lr=0.01, ),
)
# SGD
#dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
# AdamW
# dict(type='AdamW', lr=1e-4,  weight_decay=0.01)
# learning policy
lr_config = dict(policy='poly',) #dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)


