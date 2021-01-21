def _make_net(self, o):
    # Process observation
    if self.connection_type == 'ff':
        x = o
        for ilayer, hd in enumerate(self.hidden_dims):
            x = self.nonlin(U.dense(x, hd, 'l{}'.format(ilayer), U.normc_initializer(1.0)))
    else:
        raise NotImplementedError(self.connection_type)

    # Map to action
    adim, ahigh, alow = self.ac_space.shape[0], self.ac_space.high, self.ac_space.low
    assert isinstance(self.ac_bins, str)
    ac_bin_mode, ac_bin_arg = self.ac_bins.split(':')

    if ac_bin_mode == 'uniform':
        # Uniformly spaced bins, from ac_space.low to ac_space.high
        num_ac_bins = int(ac_bin_arg)
        aidx_na = bins(x, adim, num_ac_bins, 'out')  # 0 ... num_ac_bins-1
        ac_range_1a = (ahigh - alow)[None, :]
        a = 1. / (num_ac_bins - 1.) * tf.to_float(aidx_na) * ac_range_1a + alow[None, :]

    elif ac_bin_mode == 'custom':
        # Custom bins specified as a list of values from -1 to 1
        # The bins are rescaled to ac_space.low to ac_space.high
        acvals_k = np.array(list(map(float, ac_bin_arg.split(','))), dtype=np.float32)
        logger.info('Custom action values: ' + ' '.join('{:.3f}'.format(x) for x in acvals_k))
        assert acvals_k.ndim == 1 and acvals_k[0] == -1 and acvals_k[-1] == 1
        acvals_ak = (
            (ahigh - alow)[:, None] / (acvals_k[-1] - acvals_k[0]) * (acvals_k - acvals_k[0])[None, :]
            + alow[:, None]
        )

        aidx_na = bins(x, adim, len(acvals_k), 'out')  # values in [0, k-1]
        a = tf.gather_nd(
            acvals_ak,
            tf.concat(2, [
                tf.tile(np.arange(adim)[None, :, None], [tf.shape(aidx_na)[0], 1, 1]),
                tf.expand_dims(aidx_na, -1)
            ])  # (n,a,2)
        )  # (n,a)
    elif ac_bin_mode == 'continuous':
        a = U.dense(x, adim, 'out', U.normc_initializer(0.01))
    else:
        raise NotImplementedError(ac_bin_mode)

    return a

