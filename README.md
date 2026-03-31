# smollerlm-for-macplus

For those weirdos who see [llama2.c](https://github.com/karpathy/llama2.c) and think "Hmmm... what if it took 1.86 kW to run it very slowly?". Currently only tested on Apple System 6.0.8.

![](https://raw.githubusercontent.com/Daviljoe193/smollerlm-for-macplus/refs/heads/main/demoimg.png)

___

### Useage

First, you need the dependencies (In a VENV, silly c:) to create the database.

`pip install torch transformers`

And with Retro68 built and ready to use, run these commands (Substituting the toolchain file path with your actual toolchain path...

`mkdir build`
`cd build`
`cmake -DCMAKE_TOOLCHAIN_FILE='/home/dj-tst/build/Retro68/Retro68-build/toolchain/m68k-apple-macos/cmake/retro68.toolchain.cmake' ..`
`make`

And then, you need to run (Note: This will pull in my beloved [SmollerLM2-10M-sftb](https://huggingface.co/mehmetkeremturkcan/SmollerLM2-10M-sftb) from [mehmetkeremturkcan](https://huggingface.co/mehmetkeremturkcan) on HuggingFace)

`python mkmodel.py --hf mehmetkeremturkcan/SmollerLM2-10M-sftb`

This will create a sharded model, 30 shards for the slave Macs, and 2 shards + tokenizer for the master Mac. By the way, you need 31 Mac Plus's with 2.5 megabytes of RAM each.

Now you need to move a copy of the compiled application and shard to each of the slave machines (Both will fit on a floppy for each), and for the master, move the application plus the `MASTER_VOL1.BIN` `MASTER_VOL2.BIN` and `TOKEN.BIN` to the hard drive. **Make sure that the underscores are preserved, or else the models won't load!**

You'll also need a custom cable to interconnect the 31 Mac Plus's, diagram below, built with 31 Mini-DIN-8 male connectors and Cat5 ethernet cable...

![](https://raw.githubusercontent.com/Daviljoe193/smollerlm-for-macplus/refs/heads/main/cablediagram.svg)

Plug node 0 to 1, 1 to 2, until you're at node 30. Now you can finally do the dirty and run it. Just run the application on all the nodes (It'll know which node it is based on what shard[s] it has), then when they're all good, press enter on the master node, and watch as all your Mac Plus's spend an eternity ingesting the SmolLM ChatML prompt template.

That's... it, really. The default inference params are a temperature of 0.8, a top_p of 0.9, a top_k of 40, a min_p of 0.1, and a repetition penalty of 1.2 (At 10 million parameters, SmollerLM 10M really needs this).

This is where I'd say something like "You can run the dude's other models". No, that's not in the cards for this port.
