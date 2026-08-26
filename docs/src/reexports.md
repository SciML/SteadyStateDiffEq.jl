# Reexported SciML common interface

`using SteadyStateDiffEq` also brings in the parts of the SciML common interface needed
to construct and solve the problems supported by this package. These names are owned and
documented by [SciMLBase](https://docs.sciml.ai/SciMLBase/stable/):

  - Problems: `SteadyStateProblem` and `NonlinearProblem`
  - Solutions: `SteadyStateSolution` and `NonlinearSolution`
  - Solving and problem updates: `solve` and `remake`
  - Return status: `ReturnCode` and `successful_retcode`

Anything else from SciMLBase must be imported from SciMLBase directly.
