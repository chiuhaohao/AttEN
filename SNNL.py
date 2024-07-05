import numpy as np 
import torch
import torch.nn as nn
class SNNLCrossEntropy():
    STABILITY_EPS = 0.00001
    def __init__(self,
               temperature=100.,
               layer_names=None,
               factor=-10.,
               optimize_temperature=True,
               cos_distance=True):
        
        self.temperature = temperature
        self.factor = factor
        self.optimize_temperature = optimize_temperature
        self.cos_distance = cos_distance
    
    @staticmethod
    def pairwise_euclid_distance(A, B):
        """Pairwise Euclidean distance between two matrices.
        :param A: a matrix.
        :param B: a matrix.
        :returns: A tensor for the pairwise Euclidean between A and B.
        """
        batchA = A.shape[0]
        batchB = B.shape[0]

        sqr_norm_A = torch.reshape(torch.pow(A, 2).sum(axis=1), [1, batchA])
        sqr_norm_B = torch.reshape(torch.pow(B, 2).sum(axis=1), [batchB, 1])
        inner_prod = torch.matmul(B, A.T)

        tile_1 = torch.tile(sqr_norm_A, [batchB, 1])
        tile_2 = torch.tile(sqr_norm_B, [1, batchA])
        return (tile_1 + tile_2 - 2 * inner_prod)
    
    @staticmethod
    def pairwise_cos_distance(A, B):
        
        """Pairwise cosine distance between two matrices.
        :param A: a matrix.
        :param B: a matrix.
        :returns: A tensor for the pairwise cosine between A and B.
        """
        normalized_A = torch.nn.functional.normalize(A, dim=1)
        normalized_B = torch.nn.functional.normalize(B, dim=1)
        distances = 1 - torch.matmul(query_embeddings, key_embeddings.T)
        min_clip_distances = tf.math.maximum(distances, 0.0)
        return min_clip_distances
    
    @staticmethod
    def fits(A, B, temp, cos_distance):
        if cos_distance:
            distance_matrix = SNNLCrossEntropy.pairwise_cos_distance(A, B)
        else:
            distance_matrix = SNNLCrossEntropy.pairwise_euclid_distance(A, B)
            
        return torch.exp(-(distance_matrix / temp))
    
    @staticmethod
    def pick_probability(x, temp, cos_distance):
        """Row normalized exponentiated pairwise distance between all the elements
        of x. Conceptualized as the probability of sampling a neighbor point for
        every element of x, proportional to the distance between the points.
        :param x: a matrix
        :param temp: Temperature
        :cos_distance: Boolean for using cosine or euclidean distance
        :returns: A tensor for the row normalized exponentiated pairwise distance
                  between all the elements of x.
        """
        f = SNNLCrossEntropy.fits(x, x, temp, cos_distance) - torch.eye(x.shape[0]).cuda()
        return f / (SNNLCrossEntropy.STABILITY_EPS + f.sum(axis=1).unsqueeze(1))
    
    @staticmethod
    def same_label_mask(y, y2):
        """Masking matrix such that element i,j is 1 iff y[i] == y2[i].
        :param y: a list of labels
        :param y2: a list of labels
        :returns: A tensor for the masking matrix.
        """
        return (y == y2.unsqueeze(1)).squeeze().to(torch.float32)
    
    @staticmethod
    def masked_pick_probability(x, y, temp, cos_distance):
        """The pairwise sampling probabilities for the elements of x for neighbor
        points which share labels.
        :param x: a matrix
        :param y: a list of labels for each element of x
        :param temp: Temperature
        :cos_distance: Boolean for using cosine or Euclidean distance
        :returns: A tensor for the pairwise sampling probabilities.
        """
        return SNNLCrossEntropy.pick_probability(x, temp, cos_distance) * \
                                    SNNLCrossEntropy.same_label_mask(y, y)
    
    @staticmethod
    def SNNL(x, y, temp=1, cos_distance=True):
        """Soft Nearest Neighbor Loss
        :param x: a matrix.
        :param y: a list of labels for each element of x.
        :param temp: Temperature.
        :cos_distance: Boolean for using cosine or Euclidean distance.
        :returns: A tensor for the Soft Nearest Neighbor Loss of the points
                  in x with labels y.
        """
        summed_masked_pick_prob = SNNLCrossEntropy.masked_pick_probability(x, y, temp, cos_distance).sum(axis=1)
        return -torch.log(SNNLCrossEntropy.STABILITY_EPS + summed_masked_pick_prob).mean()
    
class SoftNearestNeighborLoss(nn.Module):
    def __init__(self,
               temperature=1,
               cos_distance=True):
        super(SoftNearestNeighborLoss, self).__init__()
        
        self.temperature = temperature
        self.cos_distance = cos_distance

    def pairwise_cos_distance(self, A, B):
        query_embeddings = torch.nn.functional.normalize(A, dim=1)
        key_embeddings = torch.nn.functional.normalize(B, dim=1)
        distances = 1 - torch.matmul(query_embeddings, key_embeddings.T)
        return distances

    def forward(self, embeddings, labels):
        batch_size = embeddings.shape[0]
        eps = 1e-9
        
        pairwise_dist = self.pairwise_cos_distance(embeddings, embeddings)
        pairwise_dist = pairwise_dist / self.temperature
        negexpd = torch.exp(-pairwise_dist)
        # print('negexpd: ', negexpd)

        # creating mask to sample same class neighboorhood
        pairs_y = torch.broadcast_to(labels, (batch_size, batch_size))
        mask = pairs_y == torch.transpose(pairs_y, 0, 1)
        mask = mask.float()

        # creating mask to exclude diagonal elements
        ones = torch.ones([batch_size, batch_size], dtype=torch.float32).cuda()
        dmask = ones - torch.eye(batch_size, dtype=torch.float32).cuda()

        # all class neighborhood
        alcn = torch.sum(torch.multiply(negexpd, dmask), dim=1)
        # print('alcn: ', alcn)
        # same class neighborhood
        sacn = torch.sum(torch.multiply(negexpd, mask), dim=1)
        # print('sacn: ', sacn)
        # adding eps for numerical stability
        # in case of a class having a single occurance in batch
        # the quantity inside log would have been 0
        loss = -torch.log((sacn+eps)/alcn).mean()
        # print((sacn+eps)/alcn)
        return loss
    
# class inv_SoftNearestNeighborLoss(nn.Module):
#     def __init__(self,
#                temperature=1,
#                cos_distance=True):
#         super(inv_SoftNearestNeighborLoss, self).__init__()
        
#         self.temperature = temperature
#         self.cos_distance = cos_distance

#     def pairwise_cos_distance(self, A, B):
#         query_embeddings = torch.nn.functional.normalize(A, dim=1)
#         key_embeddings = torch.nn.functional.normalize(B, dim=1)
#         distances = 1 - torch.matmul(query_embeddings, key_embeddings.T)
#         return distances

#     def forward(self, embeddings, labels):
#         batch_size = embeddings.shape[0]
#         eps = 1e-9
        
#         pairwise_dist = self.pairwise_cos_distance(embeddings, embeddings)
#         pairwise_dist = pairwise_dist / self.temperature
#         negexpd = torch.exp(-pairwise_dist)
#         # print('negexpd: ', negexpd)

#         # creating mask to sample same class neighboorhood
#         pairs_y = torch.broadcast_to(labels, (batch_size, batch_size))
#         mask = pairs_y == torch.transpose(pairs_y, 0, 1)
#         mask = mask.float()

#         # creating mask to exclude diagonal elements
#         ones = torch.ones([batch_size, batch_size], dtype=torch.float32).cuda()
#         dmask = ones - torch.eye(batch_size, dtype=torch.float32).cuda()

#         # all class neighborhood
#         alcn = torch.sum(torch.multiply(negexpd, dmask), dim=1)
#         # print('alcn: ', alcn)
#         # same class neighborhood
#         sacn = torch.sum(torch.multiply(negexpd, mask), dim=1)
#         # print('sacn: ', sacn)
#         # adding eps for numerical stability
#         # in case of a class having a single occurance in batch
#         # the quantity inside log would have been 0
#         loss = -torch.log((sacn+eps)/alcn).mean()
#         # print((sacn+eps)/alcn)
#         return loss