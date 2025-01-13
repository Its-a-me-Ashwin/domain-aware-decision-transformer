#!/usr/bin/env python3

def generate_double_inverted_pendulum_urdf(
    cart_mass=1.0,
    cart_size=(0.5, 0.3, 0.1),  # (length_x, length_y, height_z) of the cart
    link1_mass=0.5,
    link1_length=1.0,
    link1_radius=0.05,
    link2_mass=0.5,
    link2_length=1.0,
    link2_radius=0.05,
    file_path = ""
):
    """
    Generates a URDF string for a double inverted pendulum on a cart:
      - The cart can slide along x-axis (prismatic joint).
      - Two pendulum links, each rotating about y-axis (revolute joints).
      - Link inertias are automatically computed assuming solid cylinders.
      - The cart pivot is placed at z = 0.1 + link1_length + link2_length.
    
    Returns:
        A string containing the URDF.
    """

    # ------------------------------------------------------------------
    # 1) Compute the cart's vertical position
    #
    #   We want the pivot at the top of the cart to be at zPivot = 0.1 + L1 + L2.
    #   The pivot in this URDF is at the cart's top face (z = cart_size[2]/2).
    #   So the cart link origin (for visual/collision) will be placed so that
    #   pivot_z = 0.1 + link1_length + link2_length.
    #
    #   If the cart geometry is centered at (0, 0, cart_size[2]/2),
    #   the top of the cart is at z = cart_size[2].
    #
    #   We'll place the cart link's origin so that the top = zPivot.
    #
    #   So cart_origin_z = zPivot - cart_size[2].
    # ------------------------------------------------------------------
    zPivot = 0.1 + link1_length + link2_length
    cart_origin_z = zPivot - cart_size[2]

    # ------------------------------------------------------------------
    # 2) Compute inertia for each pendulum link
    #
    #   We treat each link as a solid cylinder of:
    #       length = L,
    #       radius = r,
    #       mass  = M.
    #
    #   By default in URDF, we often define the link frame at the pivot (top),
    #   and have the cylinder go from z=0 down to z=-L.
    #
    #   The inertia in the link frame typically is specified about the center
    #   of mass. For a solid cylinder aligned with z, at the center of mass:
    #       I_xx = I_zz = (1/12) * M * (3r^2 + L^2)
    #       I_yy = (1/2) * M * r^2
    #
    #   Because the COM is at z = -L/2 from the pivot, we put:
    #       <origin xyz="0 0 -L/2"/> in the <inertial>.
    #
    #   This means we do *not* need to apply the parallel-axis theorem in
    #   ixx, iyy, izz, because the inertia is specified about the COM frame.
    # ------------------------------------------------------------------

    def cylinder_inertia_xx_zz(m, r, l):
        return (1.0/12.0) * m * (3*(r**2) + (l**2))

    def cylinder_inertia_yy(m, r, l):
        return 0.5 * m * (r**2)

    # Link 1 inertia
    I1_xx = cylinder_inertia_xx_zz(link1_mass, link1_radius, link1_length)
    I1_yy = cylinder_inertia_yy(link1_mass, link1_radius, link1_length)
    I1_zz = I1_xx  # same as I_xx for a cylinder

    # Link 2 inertia
    I2_xx = cylinder_inertia_xx_zz(link2_mass, link2_radius, link2_length)
    I2_yy = cylinder_inertia_yy(link2_mass, link2_radius, link2_length)
    I2_zz = I2_xx

    # ------------------------------------------------------------------
    # 3) Build the URDF string
    # ------------------------------------------------------------------
    urdf = f"""<?xml version="1.0"?>
<robot name="double_inverted_pendulum">

  <!-- Optional "world" link -->
  <link name="world"/>

  <!-- Cart link -->
  <link name="cart_link">
    <visual>
      <origin xyz="0 0 {cart_origin_z + cart_size[2]/2}" rpy="0 0 0"/>
      <geometry>
        <box size="{cart_size[0]} {cart_size[1]} {cart_size[2]}"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 {cart_origin_z + cart_size[2]/2}" rpy="0 0 0"/>
      <geometry>
        <box size="{cart_size[0]} {cart_size[1]} {cart_size[2]}"/>
      </geometry>
    </collision>
    <inertial>
      <!-- We'll assume a simple diagonal inertia for the cart.
           Adjust or compute realistically as needed. -->
      <origin xyz="0 0 {cart_origin_z + cart_size[2]/2}" rpy="0 0 0"/>
      <mass value="{cart_mass}"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Prismatic joint (x-axis slide) between world and cart -->
  <joint name="cart_slide_joint" type="prismatic">
    <parent link="world"/>
    <child link="cart_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-5.0" upper="5.0" effort="100.0" velocity="1.0"/>
  </joint>

  <!-- First pendulum -->
  <link name="pendulum_link1">
    <visual>
      <!-- Cylinder along z, pivot at top (z=0), extends to z=-L1 -->
      <origin xyz="0 0 {-link1_length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder length="{link1_length}" radius="{link1_radius}"/>
      </geometry>
      <material name="green">
        <color rgba="0 1 0 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 {-link1_length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder length="{link1_length}" radius="{link1_radius}"/>
      </geometry>
    </collision>
    <inertial>
      <!-- Inertia specified about the COM (which is at z=-L1/2), so origin is that offset -->
      <origin xyz="0 0 {-link1_length/2}" rpy="0 0 0"/>
      <mass value="{link1_mass}"/>
      <inertia ixx="{I1_xx}" ixy="0" ixz="0" iyy="{I1_yy}" iyz="0" izz="{I1_zz}"/>
    </inertial>
  </link>

  <!-- Joint for first pendulum (revolute about y-axis) -->
  <joint name="pendulum_joint1" type="revolute">
    <parent link="cart_link"/>
    <child link="pendulum_link1"/>
    <!-- The pivot is at the top face of the cart, i.e. z = zPivot = 0.1 + L1 + L2 -->
    <origin xyz="0 0 {zPivot}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit effort="100.0" velocity="2.0" lower="-3.14159" upper="3.14159"/>
  </joint>

  <!-- Second pendulum -->
  <link name="pendulum_link2">
    <visual>
      <origin xyz="0 0 {-link2_length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder length="{link2_length}" radius="{link2_radius}"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 {-link2_length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder length="{link2_length}" radius="{link2_radius}"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 {-link2_length/2}" rpy="0 0 0"/>
      <mass value="{link2_mass}"/>
      <inertia ixx="{I2_xx}" ixy="0" ixz="0" iyy="{I2_yy}" iyz="0" izz="{I2_zz}"/>
    </inertial>
  </link>

  <!-- Joint for second pendulum (revolute about y-axis) -->
  <joint name="pendulum_joint2" type="revolute">
    <parent link="pendulum_link1"/>
    <child link="pendulum_link2"/>
    <!-- The second pendulum hangs from the bottom of link1, i.e. z=-L1 from pivot1 -->
    <origin xyz="0 0 {-link1_length}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit effort="100.0" velocity="2.0" lower="-3.14159" upper="3.14159"/>
  </joint>

</robot>
"""
    if file_path == "" :
        with open(file_path + ".urdf", "w") as f:        
            print(urdf_str, file=f)  
      
    return urdf

if __name__ == "__main__":
    # Example usage: generate URDF with default parameters
    urdf_str = generate_double_inverted_pendulum_urdf(
        cart_mass=1.0,
        cart_size=(0.5, 0.3, 0.1),
        link1_mass=0.5,
        link1_length=1.0,
        link1_radius=0.05,
        link2_mass=0.5,
        link2_length=1.0,
        link2_radius=0.05,
        file_path = "invertedPendulum"
    )