import jittor as jt
from jittor import Function
import numpy as np

class SYTgather(Function):
    def execute(self, feature, index):
        # feature [bs, numV+1, numC]
        # index [numV, 10]
        # output [bs, numV, 10, numC]
        self.save_vars = feature, index
        bs, numV, numC = feature.shape
        numV -= 1
        
        return jt.code([bs,numV,10,numC], feature.dtype, [feature,index],
            cpu_header="""
                #include <iostream>
                using namespace std;
            """,
            cpu_src='''
                for (int batchID=0; batchID < out_shape0; batchID++) {
                    for (int vID = 0; vID < out_shape1; vID++) {
                        for (int nbID = 0; nbID < out_shape2; nbID++) {
                            for (int cID = 0; cID < out_shape3; cID++) {
                                @out(batchID, vID, nbID, cID) = @in0(batchID, @in1(vID,nbID), cID);
                            }
                        }
                    }
                }
            ''')

    def grad(self, grad_x):
        feature, index = self.save_vars

        return jt.code(feature.shape, feature.dtype, [feature, index, grad_x],
            cpu_header="""
                #include <iostream>
                #include <vector>
                using namespace std;
            """,
            cpu_src='''
                vector<vector<int>> degrees(out_shape0, vector<int>(out_shape1,0));

                for (int batchID=0; batchID < out_shape0; batchID++) {
                    for (int vID = 0; vID < out_shape1; vID++) {
                        for (int cID = 0; cID < out_shape2; cID++) {
                            @out(batchID, vID, cID) = 0;
                        }
                    }
                }

                for (int batchID=0; batchID < in2_shape0; batchID++) {
                    for (int vID = 0; vID < in2_shape1; vID++) {
                        for (int nbID = 0; nbID < in2_shape2; nbID++) {
                            for (int cID = 0; cID < in2_shape3; cID++) {
                                @out(batchID, @in1(vID,nbID), cID) += @in2(batchID, vID, nbID, cID);
                                degrees[batchID][@in1(vID,nbID)] += 1;
                            }
                        }
                    }
                }

                for (int batchID=0; batchID < out_shape0; batchID++) {
                    for (int vID = 0; vID < out_shape1; vID++) {
                        if (degrees[batchID][vID] > 0) {
                            for (int cID = 0; cID < out_shape2; cID++) {
                                @out(batchID, vID, cID) = @out(batchID, vID, cID) / degrees[batchID][vID];
                            }
                        }
                    }
                }

            ''')


class binarize(Function):
    def execute(self, x):
        self.x = x
        return jt.nn.sign(x)

    def grad(self, grad):
        # print("grad:", grad.shape, grad)
        return grad


def genEdge(nb):
    '''
        generate nb (numV, 10) to Edge index (2, numEdge)
    '''
    edge_index = []
    for startID, nbIDs in enumerate(nb):
        nbIDs = nbIDs[nbIDs > 0] - 1
        edge_index += [np.array([startID, nbID]) for nbID in nbIDs]

    return np.stack(edge_index).transpose()


class printGrad(Function):
    def execute(self, x):
        return x

    def grad(self, grad):
        print("grad:", grad.shape, grad)
        return grad