import numpy as np

class MaxPool2:
    
    def iterate_regions(self, image):
        # Generates non-overlapping 2x2 image regions to pool over.
        
        h, w = image.shape
        
        for i in range(h//2):
            for j in range(w//2):
                im_region = image[(i*2):(i*2+2), (j*2):(j*2+2)]
                yield im_region, i, j
    
    def forward(self, inp):
        # Performs a forward pass of the maxpool layer using the given input.
        # Returns a 3d numpy array with dimensions (h / 2, w / 2, num_filters).
        
        self.last_input = inp
        
        h, w, num_filters = inp.shape
        output = np.zeros((h//2, w//2, num_filters))
        
        for k in range(num_filters):
            for im_region, i, j in self.iterate_regions(inp[:, :, k]):
                output[i, j, k] = np.amax(im_region, axis = (0, 1))
            
        return output
    
    def backprop(self, dl_dout):
        # Performs a backward pass of the maxpool layer.
        # Returns the loss gradient for this layer's inputs.
        
        dl_dinput = np.zeros(self.last_input.shape)
        _, _, f = self.last_input.shape
        
        for k in range(f):
            for im_region, i, j in self.iterate_regions(self.last_input[k]):
                h, w = im_region.shape
                amax = np.amax(im_region)
            
                for i2 in range(h):
                    for j2 in range(w):
                        if im_region[i2, j2] == amax:
                            dl_dinput[i * 2 + i2, j * 2 + j2, k] = dl_dout[i, j, k]
        
        return dl_dinput