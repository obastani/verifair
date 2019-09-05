from ..verify.verify import *
from ..verify.util import *
import os
import sys

# model params
cur_dir = os.path.dirname(__file__)
dis_model_path = os.path.join(cur_dir, '../../../model/dis')
gen_cat_us_path = os.path.join(cur_dir, '../../../model/generative/cat_us')
gen_cat_nonus_path = os.path.join(cur_dir, '../../../model/generative/cat_nonus')
gen_dog_us_path = os.path.join(cur_dir, '../../../model/generative/dog_us')
gen_dog_nonus_path = os.path.join(cur_dir, '../../../model/generative/dog_nonus')


BATCH_SIZE = 1000

################ helper functions for discriminative model################

cat_class_idx = 92
dog_class_idx = 164

def model_fn(features, labels, mode, params):
  """Model function for RNN classifier.
  This function sets up a neural network which applies convolutional layers (as
  configured with params.num_conv and params.conv_len) to the input.
  The output of the convolutional layers is given to LSTM layers (as configured
  with params.num_layers and params.num_nodes).
  The final state of the all LSTM layers are concatenated and fed to a fully
  connected layer to obtain the final classification scores.
  Args:
    features: dictionary with keys: inks, lengths.
    labels: one hot encoded classes
    mode: one of tf.estimator.ModeKeys.{TRAIN, INFER, EVAL}
    params: a parameter dictionary with the following keys: num_layers,
      num_nodes, batch_size, num_conv, conv_len, num_classes, learning_rate.
  Returns:
    ModelFnOps for Estimator API.
  """

  def _get_input_tensors(features, labels):
    """Converts the input dict into inks, lengths, and labels tensors."""
    # features[ink] is a sparse tensor that is [8, batch_maxlen, 3]
    # inks will be a dense tensor of [8, maxlen, 3]
    # shapes is [batchsize, 2]
    shapes = features["shape"]
    # lengths will be [batch_size]
    lengths = tf.squeeze(
        tf.slice(shapes, begin=[0, 0], size=[params.batch_size, 1]))
    inks = tf.reshape(features["ink"], [params.batch_size, -1, 3])
    if labels is not None:
      labels = tf.squeeze(labels)
    return inks, lengths, labels

  def _add_conv_layers(inks, lengths):
    """Adds convolution layers."""
    convolved = inks
    for i in range(len(params.num_conv)):
      convolved_input = convolved
      if params.batch_norm:
        convolved_input = tf.layers.batch_normalization(
            convolved_input,
            training=(mode == tf.estimator.ModeKeys.TRAIN))
      # Add dropout layer if enabled and not first convolution layer.
      if i > 0 and params.dropout:
        convolved_input = tf.layers.dropout(
            convolved_input,
            rate=params.dropout,
            training=(mode == tf.estimator.ModeKeys.TRAIN))
      convolved = tf.layers.conv1d(
          convolved_input,
          filters=params.num_conv[i],
          kernel_size=params.conv_len[i],
          activation=None,
          strides=1,
          padding="same",
          name="conv1d_%d" % i)
    return convolved, lengths

  def _add_regular_rnn_layers(convolved, lengths):
    """Adds RNN layers."""
    if params.cell_type == "lstm":
      cell = tf.nn.rnn_cell.BasicLSTMCell
    elif params.cell_type == "block_lstm":
      cell = tf.contrib.rnn.LSTMBlockCell
    cells_fw = [cell(params.num_nodes) for _ in range(params.num_layers)]
    cells_bw = [cell(params.num_nodes) for _ in range(params.num_layers)]
    if params.dropout > 0.0:
      cells_fw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_fw]
      cells_bw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_bw]
    outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        cells_fw=cells_fw,
        cells_bw=cells_bw,
        inputs=convolved,
        sequence_length=lengths,
        dtype=tf.float32,
        scope="rnn_classification")
    return outputs

  def _add_cudnn_rnn_layers(convolved):
    """Adds CUDNN LSTM layers."""
    # Convolutions output [B, L, Ch], while CudnnLSTM is time-major.
    convolved = tf.transpose(convolved, [1, 0, 2])
    lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
        num_layers=params.num_layers,
        num_units=params.num_nodes,
        dropout=params.dropout if mode == tf.estimator.ModeKeys.TRAIN else 0.0,
        direction="bidirectional")
    outputs, _ = lstm(convolved)
    # Convert back from time-major outputs to batch-major outputs.
    outputs = tf.transpose(outputs, [1, 0, 2])
    return outputs

  def _add_rnn_layers(convolved, lengths):
    """Adds recurrent neural network layers depending on the cell type."""
    if params.cell_type != "cudnn_lstm":
      outputs = _add_regular_rnn_layers(convolved, lengths)
    else:
      outputs = _add_cudnn_rnn_layers(convolved)
    # outputs is [batch_size, L, N] where L is the maximal sequence length and N
    # the number of nodes in the last layer.
    mask = tf.tile(
        tf.expand_dims(tf.sequence_mask(lengths, tf.shape(outputs)[1]), 2),
        [1, 1, tf.shape(outputs)[2]])
    zero_outside = tf.where(mask, outputs, tf.zeros_like(outputs))
    outputs = tf.reduce_sum(zero_outside, axis=1)
    return outputs

  def _add_fc_layers(final_state):
    """Adds a fully connected layer."""
    return tf.layers.dense(final_state, params.num_classes)

  # Build the model.
  inks, lengths, labels = _get_input_tensors(features, labels)
  convolved, lengths = _add_conv_layers(inks, lengths)
  final_state = _add_rnn_layers(convolved, lengths)
  logits = _add_fc_layers(final_state)
  # Compute current predictions.
  predictions = tf.argmax(logits, axis=1)
  if mode != tf.estimator.ModeKeys.PREDICT:
      # Add the loss.
      cross_entropy = tf.reduce_mean(
          tf.nn.sparse_softmax_cross_entropy_with_logits(
              labels=labels, logits=logits))
      # Add the optimizer.
      train_op = tf.contrib.layers.optimize_loss(
          loss=cross_entropy,
          global_step=tf.train.get_global_step(),
          learning_rate=params.learning_rate,
          optimizer="Adam",
          # some gradient clipping stabilizes training in the beginning.
          clip_gradients=params.gradient_clipping_norm,
          summaries=["learning_rate", "loss", "gradients", "gradient_norm"])
      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions={"logits": logits, "predictions": predictions},
          loss=cross_entropy,
          train_op=train_op,
          eval_metric_ops={"accuracy": tf.metrics.accuracy(labels, predictions)})
  else:
      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions={"logits": logits, "predictions": predictions})


def load_dis_model(path):
    model_params = tf.contrib.training.HParams(
        num_layers=3,
        num_nodes=128,
        batch_size=BATCH_SIZE,
        num_conv=[48, 64, 96],
        conv_len=[5, 5, 3],
        num_classes=345,
        learning_rate=0.0001,
        gradient_clipping_norm=9.0,
        cell_type="lstm",
        batch_norm=False,
        dropout=0.3)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    run_config = tf.estimator.RunConfig(
        model_dir=path,
        save_checkpoints_secs=300,
        save_summary_steps=100,
        session_config=sess_config)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params=model_params)
    return estimator


def get_input_fn(X, batch_size, class_idx):
  while(len(X) % batch_size != 0):
    X.append(X[-1])

  def normalize(ink):
      (min_x, max_x, min_y, max_y) = get_bounds(ink, 1)

      return [[v[0] / (max_x-min_x), v[1] / (max_y - min_y  ), v[2]] for v in ink]

  def _input_fn():
    """Estimator `input_fn`.
    Returns:
      A tuple of:
      - Dictionary of string feature name to `Tensor`.
      - `Tensor` of target labels.
    """
    features = {'ink':[], 'shape':[]}
    labels = []

    max_ink_len = 0

    for x in X:
        norm_x = normalize(x)
        norm_x = np.array(norm_x)
        features['ink'].append(norm_x)
        features['shape'].append(norm_x.shape)
        labels.append(class_idx)
        max_ink_len = max(max_ink_len, len(norm_x))

    n_inks = []
    for ink in features['ink']:
        n_ink = np.zeros((max_ink_len, 3))
        n_ink[:len(ink)] = ink
        n_inks.append(n_ink)

    features['ink'] = np.array(n_inks, dtype=np.float32)
    features['shape'] = np.array(features['shape'])

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    dataset = dataset.padded_batch(
            batch_size, padded_shapes=dataset.output_shapes)
    features, labels = dataset.make_one_shot_iterator().get_next()
    return features, labels

  return _input_fn

def predict(model, X, class_idx):
    return model.predict(input_fn = get_input_fn(X, BATCH_SIZE, class_idx))


#########################################################################

###### helper functions for generative models##############
# import magenta command line tools
from IPython.display import SVG, display
from magenta.models.sketch_rnn.sketch_rnn_train import *
from magenta.models.sketch_rnn.model import *
from magenta.models.sketch_rnn.utils import *
from magenta.models.sketch_rnn.rnn import *
import svgwrite

def load_gen_model(path):
    g = tf.Graph()
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(graph=g, config=sess_config)
    [model_params, eval_model_params, sample_model_params] = load_model(path)
    reset_graph()

    sample_model_params.batch_size = BATCH_SIZE

    with g.as_default():
        sample_model = Model(sample_model_params, reuse=tf.AUTO_REUSE)
        sess.run(tf.global_variables_initializer())
        load_checkpoint(sess, path)

    return (sample_model, sess)

# little function that displays vector images and saves them to .svg
def draw_strokes(data, factor=0.2, svg_filename = '/tmp/sketch_rnn/svg/sample.svg'):
  tf.gfile.MakeDirs(os.path.dirname(svg_filename))
  min_x, max_x, min_y, max_y = get_bounds(data, factor)
  dims = (50 + max_x - min_x, 50 + max_y - min_y)
  dwg = svgwrite.Drawing(svg_filename, size=dims)
  dwg.add(dwg.rect(insert=(0, 0), size=dims,fill='white'))
  lift_pen = 1
  abs_x = 25 - min_x
  abs_y = 25 - min_y
  p = "M%s,%s " % (abs_x, abs_y)
  command = "m"
  for i in range(len(data)):
    if (lift_pen == 1):
      command = "m"
    elif (command != "l"):
      command = "l"
    else:
      command = ""
    x = float(data[i,0])/factor
    y = float(data[i,1])/factor
    lift_pen = data[i, 2]
    p += command+str(x)+","+str(y)+" "
  the_color = "black"
  stroke_width = 1
  dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill("none"))
  dwg.save()


def sample_n(sess, model, seq_len=250, temperature=1.0, greedy_mode=False,
           z=None):
  """Samples a sequence from a pre-trained model."""

  def adjust_temp(pi_pdf, temp):
    pi_pdf = np.log(pi_pdf) / temp
    pi_pdf -= pi_pdf.max()
    pi_pdf = np.exp(pi_pdf)
    pi_pdf /= pi_pdf.sum()
    return pi_pdf

  def get_pi_idx(x, pdf, temp=1.0, greedy=False):
    """Samples from a pdf, optionally greedily."""
    if greedy:
      return np.argmax(pdf)
    pdf = adjust_temp(np.copy(pdf), temp)
    accumulate = 0
    for i in range(0, pdf.size):
      accumulate += pdf[i]
      if accumulate >= x:
        return i
    tf.logging.info('Error with sampling ensemble.')
    return -1

  def sample_gaussian_2d(mu1, mu2, s1, s2, rho, temp=1.0, greedy=False):
    if greedy:
      return mu1, mu2
    mean = [mu1, mu2]
    s1 *= temp * temp
    s2 *= temp * temp
    cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]

  prev_x = np.zeros((BATCH_SIZE, 1, 5), dtype=np.float32)
  prev_x[:, 0, 2] = 1  # initially, we want to see beginning of new stroke
  if z is None:
    z = np.random.randn(BATCH_SIZE, model.hps.z_size)  # not used if unconditional

  assert(len(z) == BATCH_SIZE)

  if not model.hps.conditional:
    prev_state = sess.run(model.initial_state)
  else:
    prev_state = sess.run(model.initial_state, feed_dict={model.batch_z: z})

  strokes = np.zeros((BATCH_SIZE, seq_len, 5), dtype=np.float32)
  mixture_params = []
  for i in range(BATCH_SIZE):
      mixture_params.append([])
  greedy = False
  temp = 1.0

  sequence_length = np.ones([BATCH_SIZE])

  for i in range(seq_len):
    if not model.hps.conditional:
      feed = {
          model.input_x: prev_x,
          model.sequence_lengths: sequence_length,
          model.initial_state: prev_state
      }
    else:
      feed = {
          model.input_x: prev_x,
          model.sequence_lengths: sequence_length,
          model.initial_state: prev_state,
          model.batch_z: z
      }

    params = sess.run([
        model.pi, model.mu1, model.mu2, model.sigma1, model.sigma2, model.corr,
        model.pen, model.final_state
    ], feed)

    [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, next_state] = params

    if i < 0:
      greedy = False
      temp = 1.0
    else:
      greedy = greedy_mode
      temp = temperature

    prev_x = np.zeros((BATCH_SIZE, 1, 5), dtype=np.float32)

    for j in range(BATCH_SIZE):

        idx = get_pi_idx(random.random(), o_pi[j], temp, greedy)

        idx_eos = get_pi_idx(random.random(), o_pen[j], temp, greedy)
        eos = [0, 0, 0]
        eos[idx_eos] = 1

        next_x1, next_x2 = sample_gaussian_2d(o_mu1[j][idx], o_mu2[j][idx],
                                              o_sigma1[j][idx], o_sigma2[j][idx],
                                              o_corr[j][idx], np.sqrt(temp), greedy)

        strokes[j, i, :] = [next_x1, next_x2, eos[0], eos[1], eos[2]]

        params = [
            o_pi[j], o_mu1[j], o_mu2[j], o_sigma1[j], o_sigma2[j], o_corr[j],
            o_pen[j]
        ]

        mixture_params[j].append(params)

        prev_x[j][0] = np.array(
            [next_x1, next_x2, eos[0], eos[1], eos[2]], dtype=np.float32)

        prev_state = next_state

  return strokes, mixture_params


def sample_drawing(dis, sess, N = BATCH_SIZE):
    def decode(z_input=None, draw_mode=True, temperature=0.1):
        z = None
        if z_input is not None:
            z = [z_input]
        sample_strokes, m = sample_n(sess, dis, 100, temperature=temperature, z=z)
        strokes = []
        for i  in range(N):
            strokes.append(to_normal_strokes(sample_strokes[i]))
        return strokes

    ret = []

    while len(ret) < N:
        for d in decode(temperature=0.5, draw_mode=False):
            ret.append(d)

    return ret[:N]


def test_gen_model(dis, sess, prefix=''):
    N = 10
    ret = sample_drawing(dis, sess, N)
    for i in range(N):
        draw_strokes(ret[i],svg_filename=prefix+'cat_'+str(i)+'.svg')



#########################################


class QDModel:

    def __init__(self, dis, sess, model, class_idx):
        self.dis = dis
        self.sess = sess
        self.model = model
        self.class_idx = class_idx
        self.num_samples = 0

    def sample(self, n_samples):
        time1 = time.time()
        X = sample_drawing(self.dis, self.sess, n_samples)
        X = X[:n_samples]
        time2 = time.time()
        Y = predict(self.model, X, self.class_idx)
        time3 = time.time()
        print('Sample time: '+str(time2-time1))
        print('Predicion time: '+str(time3-time2))
        ret = []
        for v in Y:
            pred = v['predictions']
            ret.append(int(pred == self.class_idx))
        self.num_samples += n_samples
        return ret


def main(is_cat = True, c = 0.15, Delta = 0.005, delta = 0.5*1e-5, n_max = 100000, is_causal = False, log_iters = 10):
    if is_cat:
        target = 'cat'
        us_gen_path = gen_cat_us_path
        nonus_gen_path = gen_cat_nonus_path
        class_idx = cat_class_idx
    else:
        target = 'dog'
        us_gen_path = gen_dog_us_path
        nonus_gen_path = gen_dog_nonus_path
        class_idx = dog_class_idx

    log('Verifying fairness for '+target, INFO)

    # Step 1: Load discriminative model
    log('Loading discriminative model...', INFO)
    dis_model = load_dis_model(dis_model_path)
    log('Done!', INFO)

    # Step 2: Load generative models
    log('Loading sketch-rnn models'' as generative models', INFO)

    us_model,us_sess = load_gen_model(us_gen_path)
    non_model, non_sess = load_gen_model(nonus_gen_path)

    # Test generative model
    if False:
        test_gen_model(us_model, us_sess, prefix='us')
        test_gen_model(non_model, non_sess, prefix='nonus')
    # End

    log('Done!', INFO)

    if False: #test accuracy of the classifier
        X = sample_drawing(us_model, us_sess, 1)
        dis_model.evaluate(input_fn = get_input_fn(X,8,class_idx), steps = 1)
        exit(0)

    # # Step 4: Build model
    model_us = QDModel(us_model, us_sess, dis_model, class_idx)
    model_nonus = QDModel(non_model, non_sess, dis_model, class_idx)

    runtime = time.time()

    # # Step 3: Run fairness
    result = verify(model_nonus, model_us, c, Delta, delta, BATCH_SIZE, n_max, is_causal, log_iters)

    if result is None:
        log('RESULT: Failed to converge!', INFO)
        return

    # Step 3: Post processing
    is_fair, is_ambiguous, n_successful_samples, E = result
    runtime = time.time() - runtime
    n_total_samples = model_nonus.num_samples + model_us.num_samples

    log('RESULT: Pr[fair = {}] >= 1.0 - {}'.format(is_fair, 2.0 * delta), INFO)
    log('RESULT: E[ratio] = {}'.format(E), INFO)
    log('RESULT: Is fair: {}'.format(is_fair), INFO)
    log('RESULT: Is ambiguous: {}'.format(is_ambiguous), INFO)
    log('RESULT: Successful samples: {} successful samples, Attempted samples: {}'.format(n_successful_samples, n_total_samples), INFO)
    log('RESULT: Running time: {} seconds'.format(runtime), INFO)


if __name__ == '__main__':
    # c, Delta, delta

    baseline = [[0.15, 0, 0.5*1e-5]]

    vary_delta = []

    for i in range(10):
        cur = baseline[0][:]
        cur[2] = 0.5*pow(10,-i-1)
        vary_delta.append(cur)

    # vary_Delta = []
    #
    # for i in range(5):
    #     cur = baseline[0][:]
    #     cur[0] = 0.4
    #     cur[1] = 5 * pow(10, -i-1)
    #     vary_Delta.append(cur)

    vary_c = []

    for i in range(5):
        cur = baseline[0][:]
        cur[0] = 0.4 - 0.05 + i*0.01
        vary_c.append(cur)
        cur = baseline[0][:]
        cur[0] = 0.4 + 0.05 - i*0.01
        vary_c.append(cur)

    vary_c.append([0.4, 0, 0.5*1e-5])


    settings = (
            ('baseline', baseline),
                ('vary_delta', vary_delta),
                ('vary_c', vary_c),
                # ('vary_Delta', vary_Delta)
                )

    for name, settings in settings:
        log('RESULT: running experiment '+name, INFO)
        for s in settings:
            log('RESULT: parameters delta: {}, Delta: {}, c: {}'.format(s[2]*2, s[1], s[0]), INFO)
            main(is_cat = False, c = s[0], Delta = s[1], delta = s[2], n_max = 100000, is_causal = False, log_iters = 10)
            log('\n', INFO)
            sys.stdout.flush()
        log('\n', INFO)
