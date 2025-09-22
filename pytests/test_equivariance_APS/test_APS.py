def test_APS_adapter_debug():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (1, 8, 8, 3))
    stride = (2,2)

    pX, qX, norms = polyphase_argmax(x, stride)
    print("pX,qX:", (pX,qX))

    for shift in [(0,1), (1,0), (1,1), (2,1)]:
        x_roll = jnp.roll(x, shift, axis=(1,2))
        pGX, qGX, _ = polyphase_argmax(x_roll, stride)
        print(f"shift {shift} -> pGX,qGX={pGX,qGX}")