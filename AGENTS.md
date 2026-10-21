# Steady-state mapping

Exercise scalar, dense multidimensional, and view-backed initial states when changing lowering-to-original state reconstruction. Preserve geometry and symbolic ordering without aliasing the initial state. A concrete view type cannot be reconstructed by converting a dense array to `typeof(prob.u0)`.

Keep regression coverage for the original residual, `save_idxs`, applicable `init`/`solve!` paths, and isolation across repeated cached solves. Use the official `GROUP=Core` and `GROUP=QA` groups through `Pkg.test()`. Build `docs/make.jl` when changing algorithm docstrings, and inspect the rendered contracts.
