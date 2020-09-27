import tensorflow as tf
#https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive/03_tensorflow/labs/a_tfstart.ipynb


tf.compat.v1.enable_eager_execution()

def fx(a,x):

    result = a[0] + a[1] * x + a[2] * x ** 2 + a[3] * x ** 3 + a[4] * x ** 4
    return result

def halley(i,a,x_new,x_prev):


    with tf.GradientTape() as g:
        g.watch(x_new)
        with tf.GradientTape() as gg:
            gg.watch(x_new)
            f = fx(a,x_new)
        df_dx = gg.gradient(f, x_new)
        #print("df_dx:",df_dx)
    df_dx2 = g.gradient(df_dx, x_new)
    #print("df_dx2:",df_dx2)

    numerator = 2 * f * df_dx
    denominator = 2 * (df_dx )**2 - f * df_dx2

    #new_x = x_new - (numerator/denominator)
    #x_prev = new_x

    new_x = x_new - (numerator / denominator)
    prev_x = x_new

    print("Root approximation in step {0} = {1}".format(i, new_x))

    return [i+1, a, new_x, prev_x]

#stop if |xn+1 - xn| < stop conditions
def condition(i, a, x_new, x_prev):
    variation = tf.abs(x_new - x_prev)
    return tf.less(stop_variation, variation)


a = tf.constant(
        [2.0, -4.0, 1.0, 2.0, 0.0]
    )
x = tf.constant(40.0)
x_prev = tf.constant(100.0)
i = 0

#stop_variation = 0.00001 # Variation threshold from previous iteration to stop iteration
stop_variation = 1.0

result = halley(i,a,x,x_prev)

roots =  tf.while_loop(
          condition,
          halley,
          loop_vars=[1, a, x, x_prev],
          maximum_iterations=1000)


print("Result after {0} iterations is {1}.".format(roots[0]-1, roots[2]))
