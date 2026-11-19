from helper import *
import pytest

pytestmark = pytest.mark.slow

#import os
#os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

def test_forcing_function_bvp():
    jax.config.update("jax_enable_x64", True)
    basis_degree=1
    quad_degree=4
    n=60
    length=1
    vertices_vd = np.vstack(
        [
            np.ravel(vec)
            for vec in np.meshgrid(
                np.linspace(0, length, n + 1), np.linspace(0, length, n + 1)
            )
        ],
        dtype=float,
    ).T
    cells = np.vstack(
        [
            (
                np.array([i, i + 1, i + n + 1, i + n + 2])
                if (i + 1) % (n + 1) != 0
                else np.zeros((0, 4), dtype=int)
            )
            for i in np.arange(vertices_vd.shape[0] - n - 1, dtype=int)
        ],
        dtype=int,
    )
    boundary_points = np.hstack(
        [
            np.arange(n + 1),
            vertices_vd.shape[0] - 1 - np.arange(n + 1),
            np.array(
                [[(n + 1) * i, (n + 1) * (i + 1) - 1] for i in range(1, n)]
            ).reshape(-1),
        ]
    )
    fe_type = FiniteElementType(
        cell_type=CellType.quadrilateral,
        family=ElementFamily.P,
        basis_degree=basis_degree,
        lagrange_variant=LagrangeVariant.equispaced,
        quadrature_type=QuadratureType.default,
        quadrature_degree=quad_degree,
    )    
    xi_qp,_ = fea.get_quadrature(fe_type)
    phi_qn,_ = fea.eval_basis_and_derivatives(fe_type,xi_qp)
    x_end = vertices_vd[cells,...]
    x_eqd = jnp.einsum("end,qn -> eqd", x_end, phi_qn)    
    E = cells.shape[0]  # number of elements
    M = 1
    Q = get_quadrature(fe_type=fe_type)[0].shape[0]  # number of quadrature points
    U=1
    V,D=vertices_vd.shape
    test_solution = lambda x: x[0]*x[1]*(1-x[0])*(1-x[1])
    test_rhs_hand = jax.tree_util.Partial(jax.jit(jax.vmap(lambda x: -2*jnp.ones(1,)*(x[1]*(1-x[1])+x[0]*(1-x[0])))))
    test_rhs_auto = jax.tree_util.Partial(jax.jit(jax.vmap(lambda x: jnp.ones(1,)*jnp.trace(jax.jacfwd(jax.grad(test_solution))(x)))))
    assert jnp.allclose(test_rhs_hand(vertices_vd),test_rhs_auto(vertices_vd))
    element_batches = [
        ElementBatch(
            fe_type=fe_type,
            n_dofs_per_basis=U,
            connectivity_en=cells,
            constitutive_model=fouriers_law_linearisotropic,
            material_params=jnp.ones((E,Q,M)),
            forcing_function=test_rhs_hand
        )
    ]
    dirichlet_bcs = [
        DirichletBC(index=i, component=0, value=0)
        for i in boundary_points
    ]
    
    # Boundary conditions:
    # - Fix left nodes along x-direction
    # - Fix right nodes such that the model is subjected to 1% strain along x-axis
    # - Fix bottom nodes along y-direction
    # - Fix top nodes along y-direction


    # min_xy = np.min(points, axis=0)
    # max_xy = np.max(points, axis=0)
    # left_points = np.isclose(points[:, 0], min_xy[0], atol=1e-16).nonzero()[0]
    # right_points = np.isclose(points[:, 0], max_xy[0], atol=1e-16).nonzero()[0]
    # bottom_points = np.isclose(points[:, 1], min_xy[1], atol=1e-16).nonzero()[0]
    # top_points = np.isclose(points[:, 1], max_xy[1], atol=1e-16).nonzero()[0]
    # right_u_val = (max_xy[0] - min_xy[0]) / 100.0
    # bcs = (
    #     [
    #         DirichletBC(bc_type=BCType.NODE, index=i, component=0, value=0.0)
    #         for i in left_points
    #     ]
    #     + [
    #         DirichletBC(
    #             bc_type=BCType.NODE, index=i, component=0, value=right_u_val
    #         )
    #         for i in right_points
    #     ]
    #     + [
    #         DirichletBC(bc_type=BCType.NODE, index=i, component=1, value=0.0)
    #         for i in bottom_points
    #     ]
    #     + [
    #         DirichletBC(bc_type=BCType.NODE, index=i, component=1, value=0.0)
    #         for i in top_points
    #     ]
    # )
    u_0 = jnp.zeros(shape=(V * U))

    # Solve the boundary value problem
    u,residual,element_batches = solve_bvp(
        element_residual_func=heat_conduction_residual,
        vertices_vd=vertices_vd,
        element_batches=element_batches,
        u_0_g=u_0,
        boundary_conditions=dirichlet_bcs,
        solver_options=SolverOptions(
            linear_solve_type=LinearSolverType.GMRES_JAX_SCIPY,
            linear_precond_type=PreconditionerType.JACOBI,
        ),
    )
    # result = timeit(
    #     solve_bvp,
    #     # timeit args
    #     n_calls=3,
    #     # args passed along to solve_bvp
    #     element_residual_func=linear_elasticity_residual,
    #     vertices_vd=vertices_vd,
    #     element_batches=element_batches,
    #     u_0_g=u_0,
    #     boundary_conditions=dirichlet_bcs,
    #     solver_options=SolverOptions(
    #         linear_solve_type=LinearSolverType.GMRES_JAX_SCIPY,
    #         linear_precond_type=PreconditionerType.JACOBI,
    #     ),
    # )
    # u, residual, element_batches = result[0]
    print("|R| = ", jnp.linalg.norm(residual))

    # Make sure the solution matches at the Dirichlet BCs

    assert jnp.isclose(u[boundary_points], 0).all()

test_forcing_function_bvp()