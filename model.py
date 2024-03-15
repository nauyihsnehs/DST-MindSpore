
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
#from thop import profile
import sparseconvnet as scn
#from torchstat import stat
def count_num_model_params(model):
    num = 0
    for p in list(model.trainable_params()):
        cur = 1
        for s in list(p.size()):
            cur = cur * s
        num += cur
    return num

FSIZE0 = 3
FSIZE1 = 2
class Attention(nn.Cell):
    def __init__(self):
        super(Attention, self).__init__()
        self.batchsize=8
        data_dim=3
        nf_in=1
        nf=4
        max_data_size=[128,64,64]
        #change type
        self.p0 = scn.InputLayer(data_dim, max_data_size, mode=0)
        #more feature
        self.p1=scn.SubmanifoldConvolution(data_dim, nf_in, nf, filter_size=FSIZE0, has_bias=False)
        #downsample space by factor of 2
        self.p2=scn.Sequential().add(scn.Convolution(data_dim, nf, nf, FSIZE1, 2, False))
        self.p2.add(scn.BatchNormReLU(nf))
        self.q_conv = scn.SubmanifoldConvolution(data_dim, nf, nf, filter_size=FSIZE0, has_bias=False)
        self.k_conv = scn.SubmanifoldConvolution(data_dim, nf, nf, filter_size=FSIZE0, has_bias=False)
        self.v_conv = scn.SubmanifoldConvolution(data_dim, nf, nf, filter_size=FSIZE0, has_bias=False)

        self.trans_conv = scn.SubmanifoldConvolution(data_dim, nf, nf, filter_size=FSIZE0, has_bias=False)
        self.after_normRelu = scn.BatchNormReLU(nf)
        
        self.softmax=nn.Softmax(axis=-1)

        
    def construct(self,x):
        x_p0=self.p0(x)
        x_p1=self.p1(x_p0)
        x_in=self.p2(x_p1)
        x_q=self.q_conv(x_in)
        x_k=self.k_conv(x_in)
        x_v=self.v_conv(x_in)
        batchsize=self.batchsize

        start_id=0
        x_feat=list()
        for i in range(batchsize):
            end_id=start_id+ops.sum(x_in.metadata.getSpatialLocations(x_in.spatial_size)[:,3]==i)
            dq=x_q.features[start_id:end_id, :]    #N*nf
            dk=x_k.features[start_id:end_id, :].T  #nf*N
            dv=x_v.features[start_id:end_id, :]    #N*nf
            de=ops.matmul(dq,dk)                 #N*N
            dsoft_e=self.softmax(de)
            dr=ops.matmul(dsoft_e, dv)           #N*Nf
            x_feat.append(dr)
            start_id = end_id
        x_r = ops.cat(x_feat, axis=0)           #(N1+N2..+N8)*nf
        x_r_p0=self.p0([x_in.metadata.getSpatialLocations(x_in.spatial_size),x_r])
        x_r_p0=self.trans_conv(x_r_p0)
        x_new=self.after_normRelu(x_r_p0).features
        x_r = x_in.features+x_new
        return [x_in.metadata.getSpatialLocations(x_in.spatial_size),x_r]
        
class Attention2(nn.Cell):
    def __init__(self):
        super(Attention2, self).__init__()
        # upsampled 
        nf=1
        data_dim=3
        max_data_size=[64,32,32]
        self.n0 = scn.InputLayer(data_dim, max_data_size, mode=0)
        self.n1 = scn.SubmanifoldConvolution(data_dim, nf*4, nf, filter_size=FSIZE0, has_bias=False)
        self.n2 = scn.BatchNormReLU(nf)
        self.n3 = scn.OutputLayer(data_dim)
        self.linear = nn.Dense(nf, 1)
        self.linearsdf = nn.Dense(nf, 1)

    def to_next_level_locs(self, locs, feats): # upsample factor of 2 predictions
        assert(len(locs.shape) == 2)
        data_dim = locs.shape[-1] - 1 # assumes batch mode 
        offsets = ops.nonzero(ops.ones(2,2,2)).long() # 8 x 3 
        locs_next = locs.unsqueeze(1).tile((1, 8, 1))
        locs_next[:,:,:data_dim] *= 2
        locs_next[:,:,:data_dim] += offsets
        feats_next = feats.unsqueeze(1).tile((1, 8, 1)) # TODO: CUSTOM TRILERP HERE??
        return locs_next.view(-1, locs.shape[-1]), feats_next.view(-1, feats.shape[-1])

    def construct(self,x):
        
        up_x,up_feats=self.to_next_level_locs(x[0],x[1])
        x0=self.n0([up_x,up_feats])
        x1=self.n1(x0)
        x2=self.n2(x1)
        x3=self.n3(x2)
        return x3
class SparseAttentionLayer(nn.Cell):
    def __init__(self,nf_in, nf, input_sparsetensor, return_sparsetensor, max_data_size):
        super(SparseAttentionLayer, self).__init__()
        data_dim = 3
        self.nf_in = nf_in
        self.nf = nf
        self.input_sparsetensor = input_sparsetensor
        self.return_sparsetensor = return_sparsetensor
        self.max_data_size = max_data_size
        self.batchsize = 8

        # change type
        self.p0 = scn.InputLayer(data_dim, max_data_size, mode=0)
        # downsample space by factor of 2
        self.p2 = scn.Sequential().add(scn.Convolution(data_dim, nf, nf, FSIZE1, 2, False))
        self.p2.add( scn.BatchNormReLU(nf))
        self.q_conv = scn.SubmanifoldConvolution(data_dim, nf, nf, filter_size=FSIZE0, has_bias=False)
        self.k_conv = scn.SubmanifoldConvolution(data_dim, nf, nf, filter_size=FSIZE0, has_bias=False)
        self.v_conv = scn.SubmanifoldConvolution(data_dim, nf, nf, filter_size=FSIZE0, has_bias=False)

        self.trans_conv = scn.SubmanifoldConvolution(data_dim, nf, nf, filter_size=FSIZE0, has_bias=False)
        self.after_normRelu = scn.BatchNormReLU(nf)

        self.softmax = nn.Softmax(axis=-1)
        if not self.return_sparsetensor:
            self.sparse_to_den = scn.SparseToDense(data_dim, nf)
    def construct(self, x):
        if not self.input_sparsetensor:
            x = self.p0(x)
        x_q = self.q_conv(x)
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        batchsize = 8 if x.metadata.getSpatialLocations(x.spatial_size).shape[1]>3 else 1
        start_id = 0
        x_feat = list()
        for i in range(batchsize):
            end_id = start_id + ops.sum(x.metadata.getSpatialLocations(x.spatial_size)[:, 3] == i)
            dq = x_q.features[start_id:end_id, :]  # N*nf
            dk = x_k.features[start_id:end_id, :].T  # nf*N
            dv = x_v.features[start_id:end_id, :]  # N*nf
            de = ops.matmul(dq, dk)  # N*N
            dsoft_e = self.softmax(de)
            dr = ops.matmul(dsoft_e, dv)  # N*Nf
            x_feat.append(dr)
            start_id = end_id
        x_r = ops.cat(x_feat, axis=0)  # (N1+N2..+N8)*nf
        x_r_p0 = self.p0([x.metadata.getSpatialLocations(x.spatial_size), x_r])
        x_r_trans = self.trans_conv(x_r_p0)
        x_middle_ft = x.features + self.after_normRelu(x_r_trans).features
        x_middle_sc=self.p0([x.metadata.getSpatialLocations(x.spatial_size),x_middle_ft])
        return x_middle_sc

class SparseEncoderLayer(nn.Cell):
    def __init__(self, nf_in, nf, input_sparsetensor, return_sparsetensor, max_data_size):
        super(SparseEncoderLayer, self).__init__()
        data_dim = 3
        self.nf_in = nf_in
        self.nf = nf
        self.input_sparsetensor = input_sparsetensor
        self.return_sparsetensor = return_sparsetensor
        self.max_data_size = max_data_size
        if not self.input_sparsetensor:
            self.p0 = scn.InputLayer(data_dim, self.max_data_size, mode=0)
        self.p1 = scn.SubmanifoldConvolution(data_dim, nf_in, nf, filter_size=FSIZE0, has_bias=False)
        self.p2 = scn.Sequential()
        self.p2.add(scn.ConcatTable()
                    .add(scn.Identity())
                    .add(scn.Sequential()
                        .add(scn.BatchNormReLU(nf))
                        .add(scn.SubmanifoldConvolution(data_dim, nf, nf, FSIZE0, False))
                        .add(scn.BatchNormReLU(nf))
                        .add(scn.SubmanifoldConvolution(data_dim, nf, nf, FSIZE0, False)))
                ).add(scn.AddTable())
        self.p2.add(scn.BatchNormReLU(nf))
        # downsample space by factor of 2
        self.p3 = scn.Sequential().add(scn.Convolution(data_dim, nf, nf, FSIZE1, 2, False))
        self.p3.add(scn.BatchNormReLU(nf))

        if not self.return_sparsetensor:
            self.p4 = scn.SparseToDense(data_dim, nf)

    def construct(self,x):
        if not self.input_sparsetensor:
            x = self.p0(x)
        x = self.p1(x)
        x = self.p2(x)
        ft2 = x
        x = self.p3(x)
        if self.return_sparsetensor:
            return x, [ft2]
        else: # densify
            #sparse_attention_x=self.sparse_attention(x)
            ft3=x
            x = self.p4(x)
            return x, [ft2,ft3]
class DenseEncoderLayer(nn.Cell):
    def __init__(self, nf_in, nf_out,layer):
        super(DenseEncoderLayer, self).__init__()
        self.layer=layer
        self.nf_in=nf_in
        self.nf_out=nf_out


        self.use_bias=False
        self.kernel=4 if layer<2 else 1
        self.stride=2 if layer<2 else 1
        self.padding=1 if layer<2 else 0
        self.embed_dim=nf_out
        self.num_heads=4
        self.head_dim = nf_in // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.proj1 = nn.Conv3d(nf_in, self.embed_dim, kernel_size=self.kernel, stride=self.stride, padding=self.padding,
                           has_bias=self.use_bias)
        self.q = nn.Dense(self.embed_dim,self.embed_dim,has_bias=False)
        self.kv = nn.Dense(self.embed_dim,self.embed_dim*2,has_bias=False)

        self.bn=nn.BatchNorm3d(nf_out)
        self.relu=nn.ReLU()
        self.attn_drop = nn.Dropout(0)
        self.proj = nn.Dense(self.embed_dim, self.embed_dim)
        self.proj_drop = nn.Dropout(0)
        self.bn1 = nn.BatchNorm3d(nf_out)
        self.relu1 = nn.ReLU()

        self.sr_ratio=1
        if self.sr_ratio>1:
            self.sr = nn.Conv3d(self.embed_dim, self.embed_dim, kernel_size=self.sr_ratio, stride=self.sr_ratio)
            self.norm = nn.LayerNorm(self.embed_dim)
        #add
        self.drop_path=0
        #mlp
        self.fc1 = nn.Dense(self.embed_dim, self.embed_dim)
        self.act =nn.GELU()
        self.fc2 = nn.Dense(self.embed_dim, self.embed_dim)
        self.drop = nn.Dropout(0)
    def construct(self, x):
        #patch embedding
        B,C,Z,Y,X=x.shape
        x_embed=self.proj1(x)
        x_embed = x_embed.permute(0, 2, 3, 4, 1).reshape(B, Z * Y * X//8, -1).contiguous()  # B*N*C

        #attention
        B,N,C=x_embed.shape
        q=self.q(x_embed).reshape(B,N,self.num_heads,C//self.num_heads).permute(0,2,1,3)#B*Head*N*C
        if self.sr_ratio>1:
            x_=x_embed.permute(0,2,1).reshape(B,C,Z,Y,X)
            x_=self.sr(x_).reshape(B,C,-1).permute(0,2,1)#B N1 C
            x_=self.norm(x_)
            kv=self.kv(x_).reshape(B,-1,2,self.num_heads, C // self.num_heads).permute(2,0,3,1,4)#2 B Head N1/2 C/Head
        else:
            kv = self.kv(x_embed).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,4)  # 2 B Head N1/2 C/Head
        k,v=kv[0],kv[1]#B Head N1/2 C/Head

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B Head H*W H1*W1
        attn = attn.softmax(axis=-1)
        attn = self.attn_drop(attn)

        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)  # B Head N1/2 C/Head-----B  N1/2  C
        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn)
        #mlp
        # prepare
        x=x_embed+x_attn
        #process
        x_mlp = self.fc1(x)
        x_mlp = self.act(x_mlp)
        x_mlp = self.drop(x_mlp)
        x_mlp = self.fc2(x_mlp)
        x_mlp = self.drop(x_mlp)
        x=x+x_mlp#[8,128,24] [8,16,32] batch zyx channel

        #reshape
        x=x.permute(0,2,1)
        x=x.reshape(B,-1,Z//2,Y//2,X//2)
        return x


class TSDFEncoder(nn.Cell):
    def __init__(self, nf_in, nf_per_level, nf_out, use_skip_sparse, use_skip_dense, input_volume_size):
        super(TSDFEncoder, self).__init__()
        assert (type(nf_per_level) is list)
        data_dim = 3
        self.use_skip_sparse = use_skip_sparse
        self.use_skip_dense = use_skip_dense
        #self.use_bias = True
        self.use_bias = False
        modules = []
        volume_sizes = [(np.array(input_volume_size) // (2**k)).tolist() for k in range(len(nf_per_level))]#[(np.array(input_volume_size) // (k + 1)).tolist() for k in range(len(nf_per_level))]
        print('volume_sizes',volume_sizes)
        for level in range(len(nf_per_level)):
            nf_in = nf_in if level == 0 else nf_per_level[level-1]
            input_sparsetensor = level > 0
            return_sparsetensor = (level < len(nf_per_level) - 1)
            modules.append(SparseEncoderLayer(nf_in, nf_per_level[level], input_sparsetensor, return_sparsetensor, volume_sizes[level]))

        self.process_sparse = nn.SequentialCell(*modules)
        nf = nf_per_level[-1]

        # 16 -> 8
        nf0 = nf*3 // 2
        self.encode_dense0 = nn.SequentialCell(
            nn.Conv3d(nf, nf0, kernel_size=4, stride=2, padding=1, has_bias=self.use_bias),
            nn.BatchNorm3d(nf0),
            nn.ReLU()
        )


        # 8 -> 4
        nf1 = nf*2
        self.encode_dense1 = nn.SequentialCell(
            nn.Conv3d(nf0, nf1, kernel_size=4, stride=2, padding=1, has_bias=self.use_bias),
            nn.BatchNorm3d(nf1),
            nn.ReLU()
        )
        # 4 -> 4
        nf2 = nf1
        self.bottleneck_dense2 = nn.SequentialCell(
            nn.Conv3d(nf1, nf2, kernel_size=1, has_bias=self.use_bias),
            nn.BatchNorm3d(nf2),
            nn.ReLU()
        )
        self.bottleneck_dense2_attn = nn.SequentialCell(
            nn.Conv3d(nf1, nf2, kernel_size=1, has_bias=self.use_bias),
            nn.BatchNorm3d(nf2),
            nn.ReLU()
        )
        # attention layer
        attention = []
        nf_encoder = [nf,nf0,nf1,nf2]
        for layer in range(3):
            attention_nf_in=nf_encoder[layer]
            attention_nf_out=nf_encoder[layer+1]
            attention.append(DenseEncoderLayer(attention_nf_in, attention_nf_out,layer))
        self.process_dense = nn.SequentialCell(*attention)
        # 4 -> 8
        nf3 = nf2 if not self.use_skip_dense else nf1+nf2
        nf4 = nf3 // 2
        self.decode_dense3 = nn.SequentialCell(
            nn.Conv3dTranspose(nf3, nf4, kernel_size=4, stride=2, padding=1, has_bias=self.use_bias),
            nn.BatchNorm3d(nf4),
            nn.ReLU()
        )
        self.decode_dense3_attn = nn.SequentialCell(
            nn.Conv3dTranspose(nf3, nf4, kernel_size=4, stride=2, padding=1, has_bias=self.use_bias),
            nn.BatchNorm3d(nf4),
            nn.ReLU()
        )
        # 8 -> 16
        if self.use_skip_dense:
            nf4 += nf0
        nf5 = nf4 // 2
        self.decode_dense4 = nn.SequentialCell(
            nn.Conv3dTranspose(nf4, nf5, kernel_size=4, stride=2, padding=1, has_bias=self.use_bias),
            nn.BatchNorm3d(nf5),
            nn.ReLU()
        )
        self.decode_dense4_attn = nn.SequentialCell(
            nn.Conv3dTranspose(nf4, nf5, kernel_size=4, stride=2, padding=1, has_bias=self.use_bias),
            nn.BatchNorm3d(nf5),
            nn.ReLU()
        )
        self.final = nn.SequentialCell(
            nn.Conv3d(nf5, nf_out, kernel_size=1, has_bias=self.use_bias),
            nn.BatchNorm3d(nf_out),
            nn.ReLU()
        )
        self.final_attn = nn.SequentialCell(
            nn.Conv3d(nf5, nf_out, kernel_size=1, has_bias=self.use_bias),
            nn.BatchNorm3d(nf_out),
            nn.ReLU()
        )
        self.final_all = nn.SequentialCell(
            nn.Conv3d(nf_out*2, nf_out, kernel_size=1, has_bias=self.use_bias),
            nn.BatchNorm3d(nf_out),
            nn.ReLU()
        )
        # occ prediction
        self.occpred = nn.SequentialCell(
            nn.Conv3d(nf_out, 1, kernel_size=1, has_bias=self.use_bias)
        )
        self.sdfpred = nn.SequentialCell(
            nn.Conv3d(nf_out, 1, kernel_size=1, has_bias=self.use_bias)
        )
        self.occpred_attn = nn.SequentialCell(
            nn.Conv3d(nf_out, 1, kernel_size=1, has_bias=self.use_bias)
        )
        self.sdfpred_attn = nn.SequentialCell(
            nn.Conv3d(nf_out, 1, kernel_size=1, has_bias=self.use_bias)
        )
        self.occ = nn.SequentialCell(
            nn.Conv3d(2, 1, kernel_size=1, has_bias=self.use_bias)
        )
        self.sdf = nn.SequentialCell(
            nn.Conv3d(2, 1, kernel_size=1, has_bias=self.use_bias)
        )

    def construct(self,x):
        feats_sparse = []
        for k in range(len(self.process_sparse)):
            x, ft = self.process_sparse[k](x)
            if self.use_skip_sparse:
                feats_sparse.extend(ft)
        enc0 = self.encode_dense0(x)
        enc1 = self.encode_dense1(enc0)
        bottleneck = self.bottleneck_dense2(enc1)

        enc0_attn=self.process_dense[0](x)
        enc1_attn=self.process_dense[1](enc0_attn)
        #
        bottleneck_attn = self.bottleneck_dense2_attn(enc1_attn)
        if self.use_skip_dense:
            dec0= self.decode_dense3(ops.cat([bottleneck, enc1], 1))
            dec0_attn = self.decode_dense3_attn(ops.cat([bottleneck_attn, enc1_attn], 1))
        else:
            dec0=self.decode_dense3(bottleneck)
            dec0_attn = self.decode_dense3_attn(bottleneck_attn)
        if self.use_skip_dense:
            x = self.decode_dense4(ops.cat([dec0, enc0], 1))
            x_attn = self.decode_dense4_attn(ops.cat([dec0_attn, enc0_attn], 1))
        else:
            x = self.decode_dense4(dec0)
            x_attn = self.decode_dense4_attn(dec0_attn)
        x = self.final(x)
        x_attn = self.final_attn(x_attn)
        x_all=self.final_all(ops.cat([x, x_attn], 1))
        occ = self.occpred(x_all)
        sdf = self.sdfpred(x_all)

        out = ops.cat([occ, sdf],1)
        return x, out, feats_sparse
class DenseTransformerLayer(nn.Cell):
    def __init__(self,nf_in, nf, max_data_size):
        nn.Module.__init__(self)
        data_dim=3
        self.nf_in=nf_in
        self.nf=nf
        self.max_data_size=max_data_size
        self.downsample=1
        #changetype
        self.p0 = scn.InputLayer(data_dim, max_data_size, mode=0)
        #merge connect skip信息
        self.p1 = scn.SubmanifoldConvolution(data_dim, nf_in, nf, filter_size=FSIZE0, has_bias=False)
        #attention
        self.q= scn.SubmanifoldConvolution(data_dim, nf, nf, filter_size=1, has_bias=False)
        self.k= scn.SubmanifoldConvolution(data_dim, nf, nf, filter_size=1, has_bias=False)
        self.v= scn.SubmanifoldConvolution(data_dim, nf, nf, filter_size=1, has_bias=False)
        if self.downsample==1:
            self.kv_conv = scn.Sequential().add(scn.Convolution(data_dim, nf, nf, FSIZE1, 2, False))
            self.kv_conv.add(scn.BatchNormalization(nf))
        self.softmax = nn.Softmax(axis=-1)
        self.trans_conv = scn.SubmanifoldConvolution(data_dim, nf, nf, filter_size=FSIZE0, has_bias=False)
        self.after_norm = scn.BatchNormalization(nf)
        self.after_relu = scn.ReLU()
        #mlp
        self.res=scn.Sequential()
        self.res.add(scn.ConcatTable()
                   .add(scn.Identity())
                   .add(scn.Sequential()
                        .add(scn.BatchNormReLU(nf))
                        .add(scn.SubmanifoldConvolution(data_dim, nf, nf, FSIZE0, False))
                        .add(scn.BatchNormReLU(nf))
                        .add(scn.SubmanifoldConvolution(data_dim, nf, nf, FSIZE0, False)))
                   ).add(scn.AddTable())
        self.res.add(scn.BatchNormReLU(nf))
        self.out=scn.OutputLayer(data_dim)

    def construct(self, x):
        x=self.p0(x)
        x=self.p1(x)
        x_q=self.q(x)
        kv=self.kv_conv(x)
        x_k=self.k(kv)
        x_v=self.v(kv)
        batchsize = 8 if x.metadata.getSpatialLocations(x.spatial_size).shape[1] > 3 else 1

        start_id = 0
        x_feat = list()
        for i in range(batchsize):
            end_id = start_id + ops.sum(x.metadata.getSpatialLocations(x.spatial_size)[:, 3] == i)
            dq = x_q.features[start_id:end_id, :]  # N*nf
            dk = x_k.features[start_id:end_id, :].T  # nf*N1
            dv = x_v.features[start_id:end_id, :]  # N1*nf
            de = ops.matmul(dq, dk)  # N*N1
            dsoft_e = self.softmax(de)
            dr = ops.matmul(dsoft_e, dv)  # N*nf
            x_feat.append(dr)
            start_id = end_id
        x_r = ops.cat(x_feat, axis=0)
        x_r_p0 = self.p0([x.metadata.getSpatialLocations(x.spatial_size), x_r])
        x_r_trans = self.trans_conv(x_r_p0)
        x.features = x.features + self.after_norm(x_r_trans).features
        #mlp
        x=self.res(x)
        x=self.out(x)
        return x
    
class ConnectTransformerLayer(nn.Cell):
    def __init__(self,decoder_nf_in,encoder_nf_in, nf, max_data_size):
        super(ConnectTransformerLayer, self).__init__()
        data_dim=3
        self.encoder_nf_in=encoder_nf_in
        self.decoder_nf_in=decoder_nf_in
        self.nf=nf
        self.max_data_size=max_data_size
        self.downsample=1
        # changetype
        self.p0 = scn.InputLayer(data_dim, max_data_size, mode=0)
        self.p1=scn.SubmanifoldConvolution(data_dim, decoder_nf_in, nf, filter_size=1, has_bias=False)
        #cross
        self.q = scn.SubmanifoldConvolution(data_dim, nf, nf, filter_size=1, has_bias=False)
        self.k = scn.SubmanifoldConvolution(data_dim, encoder_nf_in, nf, filter_size=1, has_bias=False)
        self.v = scn.SubmanifoldConvolution(data_dim, encoder_nf_in, nf, filter_size=1, has_bias=False)
        self.softmax = nn.Softmax(axis=-1)
        self.trans_conv = scn.SubmanifoldConvolution(data_dim, nf, nf, filter_size=1, has_bias=False)
        self.after_norm = scn.BatchNormalization(nf)
        #self
        self.q1 = scn.SubmanifoldConvolution(data_dim, nf, nf, filter_size=1, has_bias=False)
        self.k1 = scn.SubmanifoldConvolution(data_dim, nf, nf, filter_size=1, has_bias=False)
        self.v1 = scn.SubmanifoldConvolution(data_dim, nf, nf, filter_size=1, has_bias=False)
        if self.downsample == 1:
            self.kv_conv = scn.Sequential().add(scn.Convolution(data_dim, nf, nf, FSIZE1, 2, False))
            self.kv_conv.add(scn.BatchNormalization(nf))
        self.softmax1 = nn.Softmax(axis=-1)
        self.trans_conv1 = scn.SubmanifoldConvolution(data_dim, nf, nf, filter_size=FSIZE0, has_bias=False)
        self.after_norm1 = scn.BatchNormalization(nf)
        # mlp
        self.res = scn.Sequential()
        self.res.add(scn.ConcatTable()
                     .add(scn.Identity())
                     .add(scn.Sequential()
                          .add(scn.BatchNormReLU(nf))
                          .add(scn.SubmanifoldConvolution(data_dim, nf, nf, FSIZE0, False))
                          .add(scn.BatchNormReLU(nf))
                          .add(scn.SubmanifoldConvolution(data_dim, nf, nf, FSIZE0, False)))
                     ).add(scn.AddTable())
        self.res.add(scn.BatchNormReLU(nf))
        self.out = scn.OutputLayer(data_dim)

    def construct(self, x_encoder,x_decoder):
        #cross
        x_encoder=self.p0(x_encoder)

        x_decoder=self.p0(x_decoder)
        x_decoder = self.p1(x_decoder)
        x_decoder_q=self.q(x_decoder)
        x_encoder_k=self.k(x_encoder)
        x_encoder_v=self.v(x_encoder)
        batchsize = 8 if x_decoder_q.metadata.getSpatialLocations(x_decoder_q.spatial_size).shape[1] > 3 else 1

        q_start_id = 0
        kv_start_id = 0
        x_feat = list()
        for i in range(batchsize):
            q_end_id = q_start_id + ops.sum(x_decoder_q.metadata.getSpatialLocations(x_decoder_q.spatial_size)[:, 3] == i)
            kv_end_id = kv_start_id + ops.sum(x_encoder_k.metadata.getSpatialLocations(x_encoder_k.spatial_size)[:, 3] == i)
            dq = x_decoder_q.features[q_start_id:q_end_id, :]  # N*nf
            dk = x_encoder_k.features.T  # nf*N1
            dv = x_encoder_v.features # N1*nf1
            de = ops.matmul(dq, dk)  # N*N1
            print('shapes',dq.shape,dk.shape,dv.shape)
            dsoft_e = self.softmax(de)
            dr = ops.matmul(dsoft_e, dv)  # N*nf1
            x_feat.append(dr)
            q_start_id = q_end_id
            kv_start_id = kv_end_id
        x_r = ops.cat(x_feat, axis=0)
        x_r_p0 = self.p0([x_decoder_q.metadata.getSpatialLocations(x_decoder_q.spatial_size), x_r])
        x_r_trans = self.trans_conv(x_r_p0)
        # x+x_attention
        x_decoder.features = x_decoder.features + self.after_norm(x_r_trans).features

        #self
        x_decoder_q = self.q1(x_decoder)
        kv = self.kv_conv(x_decoder_q)
        x_decoder_k = self.k1(kv)
        x_decoder_v = self.v1(kv)
        start_id = 0
        x_feat = list()
        for i in range(batchsize):
            end_id = start_id + ops.sum(x_decoder.metadata.getSpatialLocations(x_decoder.spatial_size)[:, 3] == i)
            dq = x_decoder_q.features[start_id:end_id, :]  # N*nf
            dk = x_decoder_k.features[start_id:end_id, :].T  # nf*N1
            dv = x_decoder_v.features[start_id:end_id, :]  # N1*nf
            de = ops.matmul(dq, dk)  # N*N1
            dsoft_e = self.softmax1(de)
            dr = ops.matmul(dsoft_e, dv)  # N*nf
            x_feat.append(dr)
            start_id = end_id
        x_r = ops.cat(x_feat, axis=0)
        x_r_p0 = self.p0([x_decoder.metadata.getSpatialLocations(x_decoder.spatial_size), x_r])
        x_r_trans = self.trans_conv1(x_r_p0)
        # x+x_attention
        x_decoder.features = x_decoder.features + self.after_norm1(x_r_trans).features
        # mlp
        x_decoder = self.res(x_decoder)
        x_decoder = self.out(x_decoder)
        return x_decoder
class CrossTransformerLayer(nn.Cell):
    def __init__(self,decoder_nf_in,encoder_nf_in, nf, max_data_size):
        super(CrossTransformerLayer, self).__init__()
        data_dim=3
        self.encoder_nf_in=encoder_nf_in
        self.decoder_nf_in=decoder_nf_in
        self.nf=nf
        self.max_data_size=max_data_size
        self.downsample=1
        # changetype
        self.p0 = scn.InputLayer(data_dim, max_data_size, mode=0)
        self.p1=scn.SubmanifoldConvolution(data_dim, decoder_nf_in, nf, filter_size=3, has_bias=False)
        #cross
        self.q = scn.SubmanifoldConvolution(data_dim, nf, nf, filter_size=1, has_bias=False)
        self.k = scn.SubmanifoldConvolution(data_dim, encoder_nf_in, nf, filter_size=1, has_bias=False)
        self.v = scn.SubmanifoldConvolution(data_dim, encoder_nf_in, nf, filter_size=1, has_bias=False)
        self.softmax = nn.Softmax(axis=-1)
        self.trans_conv = scn.SubmanifoldConvolution(data_dim, nf, nf, filter_size=1, has_bias=False)
        self.after_norm = scn.BatchNormalization(nf)
        self.out = scn.OutputLayer(data_dim)

    def construct(self, x_encoder,x_decoder):

        #cross
        x_encoder=self.p0(x_encoder)

        x_decoder=self.p0(x_decoder)

        x_decoder = self.p1(x_decoder)
        x_decoder_q=self.q(x_decoder)
        x_encoder_k=self.k(x_encoder)
        x_encoder_v=self.v(x_encoder)
        batchsize = 8 if x_decoder_q.metadata.getSpatialLocations(x_decoder_q.spatial_size).shape[1] > 3 else 1

        q_start_id = 0
        kv_start_id = 0
        x_feat = list()
        for i in range(batchsize):
            q_end_id = q_start_id + ops.sum(x_decoder_q.metadata.getSpatialLocations(x_decoder_q.spatial_size)[:, 3] == i)
            kv_end_id = kv_start_id + ops.sum(x_encoder_k.metadata.getSpatialLocations(x_encoder_k.spatial_size)[:, 3] == i)
            dq = x_decoder_q.features[q_start_id:q_end_id, :]  # N*nf
            dk = x_encoder_k.features[kv_start_id:kv_end_id, :].T  # nf*N1
            dv = x_encoder_v.features[kv_start_id:kv_end_id, :]  # N1*nf1
            de = ops.matmul(dq, dk)  # N*N1
            dsoft_e = self.softmax(de)
            dr = ops.matmul(dsoft_e, dv)  # N*nf1
            x_feat.append(dr)
            q_start_id = q_end_id
            kv_start_id = kv_end_id
        x_r = ops.cat(x_feat, axis=0)
        x_r_p0 = self.p0([x_decoder_q.metadata.getSpatialLocations(x_decoder_q.spatial_size), x_r])
        x_r_trans = self.trans_conv(x_r_p0)
        # x+x_attention
        x_decoder.features = x_decoder.features + self.after_norm(x_r_trans).features
        x_decoder=self.out(x_decoder)
        return x_decoder

class SelfTransformerLayer(nn.Cell):
    def __init__(self,nf_in, nf, max_data_size):
        super(SelfTransformerLayer, self).__init__()
        data_dim=3
        self.nf_in=nf_in
        self.nf=nf
        self.max_data_size=max_data_size
        self.downsample=1
        #changetype
        self.p0 = scn.InputLayer(data_dim, max_data_size, mode=0)
        #merge connect skip信息
        self.p1 = scn.SubmanifoldConvolution(data_dim, nf_in, nf, filter_size=FSIZE0, has_bias=False)
        #attention
        self.q= scn.SubmanifoldConvolution(data_dim, nf, nf, filter_size=1, has_bias=False)
        self.k= scn.SubmanifoldConvolution(data_dim, nf, nf, filter_size=1, has_bias=False)
        self.v= scn.SubmanifoldConvolution(data_dim, nf, nf, filter_size=1, has_bias=False)
        if self.downsample==1:
            self.kv_conv = scn.Sequential().add(scn.Convolution(data_dim, nf, nf, FSIZE1, 2, False))
            self.kv_conv.add(scn.BatchNormalization(nf))
        self.softmax = nn.Softmax(axis=-1)
        self.trans_conv = scn.SubmanifoldConvolution(data_dim, nf, nf, filter_size=FSIZE0, has_bias=False)
        self.after_norm = scn.BatchNormalization(nf)
        self.after_relu = scn.ReLU()
        #mlp
        self.res=scn.Sequential()
        self.res.add(scn.ConcatTable()
                   .add(scn.Identity())
                   .add(scn.Sequential()
                        .add(scn.BatchNormReLU(nf))
                        .add(scn.SubmanifoldConvolution(data_dim, nf, nf, FSIZE0, False))
                        .add(scn.BatchNormReLU(nf))
                        .add(scn.SubmanifoldConvolution(data_dim, nf, nf, FSIZE0, False)))
                   ).add(scn.AddTable())
        self.res.add(scn.BatchNormReLU(nf))
        self.out=scn.OutputLayer(data_dim)

    def construct(self, x):
        x=self.p0(x)
        x=self.p1(x)

        x_q=self.q(x)
        kv=self.kv_conv(x)
        x_k=self.k(kv)
        x_v=self.v(kv)
        batchsize = 8 if x.metadata.getSpatialLocations(x.spatial_size).shape[1] > 3 else 1
        start_id = 0
        start_id_kv=0
        x_feat = list()
        for i in range(batchsize):
            end_id = start_id + ops.sum(x.metadata.getSpatialLocations(x.spatial_size)[:, 3] == i)
            end_id_kv = start_id_kv + ops.sum(x_k.metadata.getSpatialLocations(x_k.spatial_size)[:, 3] == i)
            dq = x_q.features[start_id:end_id, :]  # N*nf
            dk = x_k.features[start_id_kv:end_id_kv, :].T  # nf*N1
            dk = x_k.features.T  # nf*N1
            dv = x_v.features[start_id_kv:end_id_kv, :]  # N1*nf
            dv = x_v.features # N1*nf
            de = ops.matmul(dq, dk)  # N*N1
            dsoft_e = self.softmax(de)
            dr = ops.matmul(dsoft_e, dv)  # N*nf
            x_feat.append(dr)
            start_id = end_id
            start_id_kv = end_id_kv
        x_r = ops.cat(x_feat, axis=0)
        x_r_p0 = self.p0([x.metadata.getSpatialLocations(x.spatial_size), x_r])
        x_r_trans = self.trans_conv(x_r_p0)
        #x+x_attention
        x.features = x.features + self.after_norm(x_r_trans).features
        #mlp
        x=self.res(x)
        x=self.out(x)
        return x
class Refinement(nn.Cell):
    def __init__(self, nf_in, nf, pass_occ, pass_feats, max_data_size, truncation=3):
        super(Refinement, self).__init__()
        data_dim = 3
        self.pass_occ = pass_occ
        self.pass_feats = pass_feats
        self.nf_in = nf_in
        self.nf = nf
        self.truncation = truncation
        self.max_data_size=max_data_size
        self.p0 = scn.InputLayer(data_dim, max_data_size, mode=0)
        self.p1 = scn.SubmanifoldConvolution(data_dim, nf_in, nf, filter_size=FSIZE0, has_bias=False)
        self.p2 = scn.FullyConvolutionalNet(data_dim, reps=1, nPlanes=[nf, nf, nf], residual_blocks=True)
        if max_data_size[0]==32:
            self.self_transformer=SelfTransformerLayer(nf_in,nf,max_data_size)
        if max_data_size[0]==64:
            self.cross_transformer=CrossTransformerLayer(18,16,18,max_data_size)
            self.self_transformer = SelfTransformerLayer(nf_in, nf, max_data_size)
        self.p3 = scn.BatchNormReLU(nf*3)
        self.p4 = scn.OutputLayer(data_dim)

        # upsampled 
        self.n0 = scn.InputLayer(data_dim, max_data_size, mode=0)
        if  max_data_size[0]==32 or max_data_size[0]==64:
            self.n1 = scn.SubmanifoldConvolution(data_dim, nf, nf, filter_size=FSIZE0, has_bias=False)
            self.n2 = scn.BatchNormReLU(nf)
        else:
            self.n1 = scn.SubmanifoldConvolution(data_dim, nf*3, nf, filter_size=FSIZE0, has_bias=False)
            self.n2 = scn.BatchNormReLU(nf)
        self.n3 = scn.OutputLayer(data_dim)
        self.linear = nn.Dense(nf, 1)
        self.linearsdf = nn.Dense(nf, 1)
    def concat_skip(self, x_from, x_to, spatial_size, batch_size):
        locs_from = x_from[0]
        locs_to = x_to[0]
        if len(locs_from) == 0 or len(locs_to) == 0:
            return x_to
        # python implementation here
        locs_from = (locs_from[:,0] * spatial_size[1] * spatial_size[2] + locs_from[:,1] * spatial_size[2] + locs_from[:,2]) * batch_size + locs_from[:,3]
        locs_to = (locs_to[:,0] * spatial_size[1] * spatial_size[2] + locs_to[:,1] * spatial_size[2] + locs_to[:,2]) * batch_size + locs_to[:,3]
        indicator_from = ops.zeros(spatial_size[0]*spatial_size[1]*spatial_size[2]*batch_size, dtype=mindspore.Tensor.long)
        indicator_to = ops.zeros(spatial_size[0]*spatial_size[1]*spatial_size[2]*batch_size, dtype=mindspore.Tensor.long)
        indicator_from[locs_from] = ops.arange(locs_from.shape[0], device=locs_from.device) + 1
        indicator_to[locs_to] = ops.arange(locs_to.shape[0], device=locs_to.device) + 1
        inds = ops.nonzero((indicator_from > 0) & (indicator_to > 0)).squeeze()
        feats_from = x_from[1].new_zeros(x_to[1].shape[0], x_from[1].shape[1])
        if inds.shape[0] > 0:
            feats_from[indicator_to[inds]-1] = x_from[1][indicator_from[inds]-1]
        x_to[1] = ops.cat([x_to[1], feats_from], 1)
        return x_to
    def to_next_level_locs(self, locs, feats): # upsample factor of 2 predictions
        assert(len(locs.shape) == 2)
        data_dim = locs.shape[-1] - 1 # assumes batch mode 
        offsets = ops.nonzero(ops.ones(2,2,2)).long() # 8 x 3 
        locs_next = locs.unsqueeze(1).tile((1, 8, 1))
        locs_next[:,:,:data_dim] *= 2
        locs_next[:,:,:data_dim] += offsets
        feats_next = feats.unsqueeze(1).tile((1, 8, 1)) # TODO: CUSTOM TRILERP HERE
        return locs_next.view(-1, locs.shape[-1]), feats_next.view(-1, feats.shape[-1])

    def construct(self,x,h):
        if h==0:
            input_locs=x[1][0]
        else:
            input_locs = x[1][0]
        if len(input_locs) == 0:
            return [[],[]],[[],[]]
        if self.max_data_size[0]==32 :
            x=self.concat_skip(x[0][0],x[1],x[0][1],x[2])
            x=self.self_transformer(x)
        elif self.max_data_size[0]==64:
            #encoder_ft0 decoder_ft1
            x1=self.cross_transformer(x[3][0],x[1])
            #encoder_ft1 decoder_ft1+encoder_ft0 max_data_size batch_size
            x=self.concat_skip(x[0][0],[input_locs,x1],x[0][1],x[2])
            x=self.self_transformer(x)
        else:
            x=self.concat_skip(x[0][0],x[1],x[0][1],x[2])
            x = self.p0(x)
            x = self.p1(x)
            x = self.p2(x)

            x = self.p3(x)
            x = self.p4(x)
        
        locs_unfilt, feats = self.to_next_level_locs(input_locs, x)

        x = self.n0([locs_unfilt, feats])
        x = self.n1(x)
        x = self.n2(x)
        x = self.n3(x)

        # predict occupancy
        out = self.linear(x)
        sdf = self.linearsdf(x)
        # mask out for next level processing
        if h==0:
            mask = (nn.Sigmoid()(out) > 0.4).view(-1)
        else:
            mask = (nn.Sigmoid()(out) > 0.4).view(-1)
        locs = locs_unfilt[mask]
        
        out = ops.cat([out, sdf],1)
        if self.pass_feats and self.pass_occ:
            feats = ops.cat([x[mask], out[mask]], 1)
        elif self.pass_feats:
            feats = x[mask]
        elif self.pass_occ:
            feats = out[mask]
        return [locs, feats], [locs_unfilt, out]

class SurfacePrediction(nn.Cell):
    def __init__(self, nf_in, nf, nf_out, max_data_size):
        super(SurfacePrediction, self).__init__()
        data_dim = 3
        self.p0 = scn.InputLayer(data_dim, max_data_size, mode=0)
        self.p1 = scn.SubmanifoldConvolution(data_dim, nf_in, nf, filter_size=FSIZE0, has_bias=False)
        self.p2 = scn.Sequential()
        self.p2.add(scn.ConcatTable()
                    .add(scn.Identity())
                    .add(scn.Sequential()
                         .add(scn.BatchNormReLU(nf))
                         .add(scn.SubmanifoldConvolution(data_dim, nf, nf, FSIZE0, False))
                         .add(scn.BatchNormReLU(nf))
                         .add(scn.SubmanifoldConvolution(data_dim, nf, nf, FSIZE0, False)))
                    ).add(scn.AddTable())
        self.p3 = scn.BatchNormReLU(nf)
        self.p4 = scn.OutputLayer(data_dim)
        self.linear = nn.Dense(nf, nf_out)
    def construct(self,x):
        if len(x[0]) == 0:
            return [], []

        x = self.p0(x)
        x = self.p1(x)
        x = self.p2(x)
        x = self.p3(x)
        x = self.p4(x)
        
        x=self.linear(x)
        return x


# ==== model ==== #
class GenModel(nn.Cell):
    def __init__(self, encoder_dim, input_dim, input_nf, nf_coarse, nf, num_hierarchy_levels, pass_occ, pass_feats, use_skip_sparse, use_skip_dense, truncation=3):
        super(GenModel, self).__init__()
        self.truncation = truncation
        self.pass_occ = pass_occ
        self.pass_feats = pass_feats
        # encoder
        if not isinstance(input_dim, (list, tuple, np.ndarray)):
            print('1 to 3')
            input_dim = [input_dim, input_dim, input_dim]
        self.nf_per_level = [int(encoder_dim*(1+float(k)/(num_hierarchy_levels-2))) for k in range(num_hierarchy_levels-1)] if num_hierarchy_levels > 2 else [encoder_dim]*(num_hierarchy_levels-1)
        self.use_skip_sparse = use_skip_sparse
        self.encoder = TSDFEncoder(input_nf, self.nf_per_level, nf_coarse, self.use_skip_sparse, use_skip_dense, input_volume_size=input_dim)
        self.refine_sizes = [(np.array(input_dim) // (pow(2,k))).tolist() for k in range(num_hierarchy_levels-1)][::-1]
        self.nf_per_level.append(self.nf_per_level[-1])
        print('#params encoder', count_num_model_params(self.encoder))

        # sparse prediction
        self.data_dim = 3
        self.refinement = scn.Sequential()
        for h in range(1, num_hierarchy_levels):
            nf_in = 0 if not self.use_skip_sparse else self.nf_per_level[num_hierarchy_levels - h]
            if pass_occ:
                nf_in += 2
            if pass_feats:
                nf_in += (nf_coarse if h == 1 else nf)
            print('refine',nf_in,nf)
            self.refinement.add(Refinement(nf_in, nf, pass_occ, pass_feats, self.refine_sizes[h-1], truncation=self.truncation))
        print('#params refinement', count_num_model_params(self.refinement))
        self.PRED_SURF = True
        if self.PRED_SURF:
            nf_in = 0 if not self.use_skip_sparse else self.nf_per_level[0]
            nf_out = 1
            if pass_occ:
                nf_in += 2
            if pass_feats:
                nf_in += nf
            print('pred', nf_in, nf)
            self.surfacepred = SurfacePrediction(nf_in, nf, nf_out, self.refine_sizes[-1])
            print('#params surfacepred', count_num_model_params(self.surfacepred))
    def dense_coarse_to_sparse(self, coarse_feats, coarse_occ, truncation):
        nf = coarse_feats.shape[1]
        batch_size = coarse_feats.shape[0]
        # sparse locations
        locs_unfilt = ops.nonzero(ops.ones([coarse_occ.shape[2], coarse_occ.shape[3], coarse_occ.shape[4]])).unsqueeze(0).tile((coarse_occ.shape[0], 1, 1)).view(-1, 3)
        batches = ops.arange(coarse_occ.shape[0]).to(locs_unfilt.device).unsqueeze(1).tile((1, coarse_occ.shape[2]*coarse_occ.shape[3]*coarse_occ.shape[4])).view(-1, 1)
        locs_unfilt = ops.cat([locs_unfilt, batches], 1)
        mask = nn.Sigmoid()(coarse_occ[:,0,:,:,:]) > 0
        if self.pass_feats:
            feats_feats = coarse_feats.permute(0, 2, 3, 4, 1).contiguous().view(batch_size, -1, nf)
            feats_feats = feats_feats[mask.view(batch_size, -1)]
        coarse_occ = coarse_occ.permute(0, 2, 3, 4, 1).contiguous()
        if self.pass_occ:
            occ_feats = coarse_occ[mask]
        if self.pass_occ and self.pass_feats:
            feats = ops.cat([occ_feats, feats_feats], 1)
        elif self.pass_occ:
            feats = occ_feats
        elif self.pass_feats:
            feats = feats_feats
        locs = locs_unfilt[mask.view(-1)]
        return locs, feats, [locs_unfilt, coarse_occ.view(-1, 2)]

    def concat_skip(self, x_from, x_to, spatial_size, batch_size):
        locs_from = x_from[0]
        locs_to = x_to[0]
        if len(locs_from) == 0 or len(locs_to) == 0:
            return x_to
        # python implementation here
        locs_from = (locs_from[:,0] * spatial_size[1] * spatial_size[2] + locs_from[:,1] * spatial_size[2] + locs_from[:,2]) * batch_size + locs_from[:,3]
        locs_to = (locs_to[:,0] * spatial_size[1] * spatial_size[2] + locs_to[:,1] * spatial_size[2] + locs_to[:,2]) * batch_size + locs_to[:,3]
        indicator_from = ops.zeros(spatial_size[0]*spatial_size[1]*spatial_size[2]*batch_size, dtype=torch.long, device=locs_from.device)
        indicator_to = indicator_from.clone()
        indicator_from[locs_from] = ops.arange(locs_from.shape[0], device=locs_from.device) + 1
        indicator_to[locs_to] = ops.arange(locs_to.shape[0], device=locs_to.device) + 1
        inds = ops.nonzero((indicator_from > 0) & (indicator_to > 0)).squeeze()
        feats_from = x_from[1].new_zeros(x_to[1].shape[0], x_from[1].shape[1])
        if inds.shape[0] > 0:
            feats_from[indicator_to[inds]-1] = x_from[1][indicator_from[inds]-1]
        x_to[1] = ops.cat([x_to[1], feats_from], 1)
        return x_to

    def update_sizes(self, input_max_dim, refine_max_dim):
        print('[model:update_sizes]', input_max_dim, refine_max_dim)
        if not isinstance(input_max_dim, (list, tuple, np.ndarray)):
            input_max_dim = [input_max_dim, input_max_dim, input_max_dim]
            print('change type')
        if not isinstance(refine_max_dim, (list, tuple, np.ndarray)):
            refine_max_dim = [refine_max_dim, refine_max_dim, refine_max_dim]
        for k in range(3):
            print(' input_max_dim', input_max_dim)
            self.encoder.process_sparse[0].p0.spatial_size[k] = input_max_dim[k]
            #self.encoder.process_sparse[2].sparse_attention.p0.spatial_size[k]=input_max_dim[k]/8
            for h in range(len(self.refinement)):
                self.refinement[h].p0.spatial_size[k] = refine_max_dim[k]
                refine_max_dim[k] *= 2
                self.refinement[h].n0.spatial_size[k] = refine_max_dim[k]
            self.surfacepred.p0.spatial_size[k] = refine_max_dim[k]
    def construct(self, x, loss_weights):
        outputs = []
        # encode
        x, out, feats_sparse = self.encoder(x)
        batch_size = x.shape[0]
        if self.use_skip_sparse:
            for k in range(len(feats_sparse)):
                feats_sparse[k] = ([feats_sparse[k].metadata.getSpatialLocations(feats_sparse[k].spatial_size), scn.OutputLayer(3)(feats_sparse[k])], feats_sparse[k].spatial_size)
        locs, feats, out = self.dense_coarse_to_sparse(x, out, truncation=3)
        outputs.append(out)

        x_sparse = [locs, feats]
        for h in range(len(self.refinement)):
            if loss_weights[h+1] > 0:
                if self.use_skip_sparse:
                    if h==0:
                        x_sparse=[feats_sparse[len(self.refinement)-h],x_sparse,batch_size]
                    else:
                        x_sparse=[feats_sparse[len(self.refinement)-h],x_sparse,batch_size,feats_sparse[len(self.refinement)-h+1]]

                x_sparse, occ = self.refinement[h](x_sparse,h)
                outputs.append(occ)
            else:
                outputs.append([[],[]])
        # surface prediction
        locs = x_sparse[0]
        if self.PRED_SURF and loss_weights[-1] > 0:
            if self.use_skip_sparse:
                x_sparse = self.concat_skip(feats_sparse[0][0], x_sparse, feats_sparse[0][1], batch_size)
            x_sparse = self.surfacepred(x_sparse)
            return [locs, x_sparse], outputs
        return [[],[]], outputs


if __name__ == '__main__':
    use_cuda = True
    batch_size=8
    model = GenModel(encoder_dim=8, input_dim=(128,64,64), input_nf=1, nf_coarse=16, nf=16, num_hierarchy_levels=4, pass_occ=True, pass_feats=True, use_skip_sparse=1, use_skip_dense=1)
    # batch size 10 -- all batches are identical
    locs = ops.randint(low=0, high=64, size=(200000, 3)).unsqueeze(0).tile((batch_size, 1, 1)).view(-1, 3)
    locs[:,0]*=2
    batches = ops.ones(locs.shape[0]).long()
    for b in range(batch_size):
        batches[b*200000:(b+1)*200000] = b
    batches = batches.unsqueeze(1)
    locs = ops.cat([locs, batches], 1)
    feats = ops.rand(200000).unsqueeze(0).tile(batch_size, 1).view(-1, 1).float()
    print('locs', locs.shape, ops.min(locs).item(), ops.max(locs).item())
    print('feats', feats.shape, ops.min(feats).item(), ops.max(feats).item())
    print(locs,feats)
    if use_cuda:
        model = model.cuda()
        locs = locs.cuda()
        feats = feats.cuda()

