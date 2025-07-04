package neural.networks;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class NMaths {

    public static INDArray leakyReLU(INDArray x) {
        return Transforms.max(x, x.mul(0.01)); // max(x, 0.01x)
    }

    public static INDArray derivativeLeakyReLU(INDArray x) {
        INDArray derivative = x.gt(0).castTo(x.dataType());
        INDArray leakyMask = x.lte(0).castTo(x.dataType()).mul(0.01);
        return derivative.add(leakyMask);
    }    

}
