"elements.py"


import math
import numpy as np

from mzm_model.core.emulation_settings import h, frequency, q, eta_s, input_current, v_pi, phi_laser, gamma_1, \
    k_splitter, gamma_2, v_off, insertion_loss, er

from mzm_model.core.math_utils import lin2db

'''Ideal light source'''


class IdealLightSource(object):
    def __init__(self, current):
        self._current = current
        self._input_power = 0
        self._out_field = 0

    @property
    def current(self):
        return self._current

    @property
    def input_power(self):
        return self._input_power

    @input_power.setter
    def input_power(self, input_power):
        self._input_power = input_power

    @property
    def out_field(self):
        return self._out_field

    @out_field.setter
    def out_field(self, out_field):
        self._out_field = out_field

    '''Ideal Light Source methods'''

    def calculate_alfa_s(self):
        alfa_s = ((h * frequency) / q) * eta_s  # [V]
        return alfa_s

    def calculate_optical_input_power(self, input_source):
        alfa = self.calculate_alfa_s()
        optical_input_power = alfa * input_source  # [W]
        self.input_power = optical_input_power
        self.out_field = np.sqrt(optical_input_power) * np.exp(phi_laser * -2j)
        return optical_input_power


'''Splitter'''


class Splitter(object):
    def __init__(self, input_power, input_field):
        self._input_power = input_power
        self._input_field = input_field
        self._A_in_field = 0
        self._B_in_field = 0

    @property
    def input_power(self):
        return self._input_power

    @property
    def input_field(self):
        return self._input_field

    @property
    def A_in_field(self):
        return self._A_in_field

    @A_in_field.setter
    def A_in_field(self, A_in_field):
        self._A_in_field = A_in_field

    @property
    def B_in_field(self):
        return self._B_in_field

    @B_in_field.setter
    def B_in_field(self, B_in_field):
        self._B_in_field = B_in_field

    '''Splitter methods'''

    def calculate_arms_input_fields(self):
        A_in_power = self.input_power * (gamma_1 ** 2)
        B_in_power = self.input_power * (1 - (gamma_1 ** 2))
        a_amplitude = (np.sqrt(A_in_power)) / (np.sqrt(k_splitter))
        b_amplitude = (np.sqrt(B_in_power)) / (np.sqrt(k_splitter))
        A_in_field = self.input_field * gamma_1
        B_in_field = (self.input_field * gamma_1) * 1j
        return A_in_field, B_in_field


'''Waveguide'''


class Waveguide(object):
    def __init__(self, in_field):
        self._in_field = in_field
        self._out_field = 0

    @property
    def in_field(self):
        return self._in_field

    @property
    def out_field(self):
        return self._out_field

    @out_field.setter
    def out_field(self, out_field):
        self._out_field = out_field

    '''Waveguides methods'''

    def delta_phi_evaluation(self, v_arm, v_pi):
        delta_phi = np.pi*(v_arm/v_pi)
        return delta_phi

    def delta_phi_evaluation_k(self, v_arm, k_arm):
        delta_phi = v_arm*k_arm
        return delta_phi

    def out_field_evaluation(self, arm_field, v_arm, v_pi):
        delta_phi = self.delta_phi_evaluation(v_arm, v_pi)
        out_field = arm_field*np.exp(delta_phi*1j)
        return out_field


'''Combiner'''


class Combiner(object):
    def __init__(self, in_A, in_B):
        self._in_A = in_A
        self._in_B = in_B
        self._out_field = 0

    @property
    def in_A(self):
        return self._in_A

    @property
    def in_B(self):
        return self._in_B

    @property
    def out_field(self):
        return self._out_field
    
    @out_field.setter
    def out_field(self, out_field):
        self._out_field = out_field

    '''Combiner methods'''

    def combiner_out_field(self, in_A, in_B):
        comb_out_field = gamma_2*(in_A+in_B*1j)
        self.out_field = comb_out_field
        return comb_out_field


'''Ideal MZM'''


class IdealMZM(object):
    def __init__(self, v_cm, v_diff, v_pi):
        self._v_cm = v_cm
        self._v_diff = v_diff
        self._v_pi = v_pi

    @property
    def v_cm(self):
        return self._v_cm

    @property
    def v_diff(self):
        return self._v_diff

    @property
    def v_pi(self):
        return self._v_pi

    '''Ideal MZM methods'''

    def field_ratio_evaluation(self):
        v_cm = self.v_cm
        v_diff = self.v_diff
        v_pi = self.v_pi
        field_ratio = np.exp(((np.pi * v_cm) / (2 * v_pi)) * 1j) * np.sin((np.pi / 2) * (v_diff / v_pi))
        return field_ratio

    # define electro-optic transfer function
    def eo_tf_evaluation(self):
        v_cm = self.v_cm
        v_diff = self.v_diff
        v_pi = self.v_pi
        eo_tf = (np.sin((np.pi / 2) * (v_diff / v_pi))) ** 2
        return eo_tf

    from mzm_model.core.utils import eo_tf_draw


'''Single Electrode MZM'''


class MZMSingleElectrode(object):
    def __init__(self, in_field, v_in, v_pi):
        self._in_field = in_field
        self._v_in = v_in
        self._v_pi = v_pi
        self._out_field = 0

    @property
    def in_field(self):
        return self._in_field

    @property
    def v_in(self):
        return self._v_in

    @property
    def v_pi(self):
        return self._v_pi

    @property
    def out_field(self):
        return self._out_field

    @out_field.setter
    def out_field(self, out_field):
        self._out_field = out_field

    '''Single Electrode MZM methods'''

    def out_mzm_field_evaluation(self):
        v_pi = self.v_pi
        in_field = self.in_field
        v_in = self.v_in
        arg = (np.pi / 2) * ((v_in - v_off) / v_pi)
        # out_field = in_field*(np.sin(arg) - np.cos(arg)*(1/np.sqrt(er))*1j)
        out_field = in_field * np.sqrt(insertion_loss) * (np.sin(arg) - ((1 / np.sqrt(er)) * np.cos(arg)) * 1j)
        self.out_field = out_field
        return out_field

    from mzm_model.core.utils import eo_tf_draw


'''Non-ideal MZM'''


class NonIdealMZM(object):
    def __init__(self, v_cm, v_diff, v_pi, er, insertion_loss, phase_offset, v_off):
        self._v_cm = v_cm
        self._v_diff = v_diff
        self._v_pi = v_pi
        self._er = er
        self._insertion_loss = insertion_loss
        self._phase_offset = phase_offset
        self._v_off = v_off

    @property
    def v_cm(self):
        return self._v_cm

    @property
    def v_diff(self):
        return self._v_diff

    @property
    def v_pi(self):
        return self._v_pi

    @property
    def er(self):
        return self._er

    @property
    def insertion_loss(self):
        return self._insertion_loss

    @property
    def phase_offset(self):
        return self._phase_offset

    @property
    def v_off(self):
        return self._v_off

    '''Non-Ideal MZM methods'''

    def field_ratio_evaluation_non_ideal(self):
        v_cm = self.v_cm
        v_diff = self.v_diff
        v_pi = self.v_pi
        er = self.er
        v_off = self.v_off
        insertion_loss = self.insertion_loss
        phase_offset = self.phase_offset
        field_ratio = np.sqrt(insertion_loss) * np.exp(((np.pi * v_cm) / (2 * v_pi)) * 1j) * \
                    (np.sin((np.pi / 2) * ((v_diff - v_off) / v_pi)) - (1 / np.sqrt(er)) * 1j * np.cos((np.pi / 2) * \
                    ((v_diff - v_off) / v_pi)))
        return field_ratio

    # define electro-optic transfer function
    def eo_tf_evaluation_non_ideal(self):
        v_cm = self.v_cm
        v_diff = self.v_diff
        v_pi = self.v_pi
        er = self.er
        v_off = self.v_off
        insertion_loss = self.insertion_loss
        phase_offset = self.phase_offset
        eo_tf = insertion_loss * (
                    (np.sin((np.pi / 2) * ((v_diff - v_off) / v_pi))) ** 2 + (1 / er) * (np.cos((np.pi / 2) * \
                    ((v_diff - v_off) / v_pi))) ** 2)
        return eo_tf

    from mzm_model.core.utils import eo_tf_draw


'''InP MZM'''


class GriffinMZM(object):
    def __init__(self, v_bias, v_A, v_B, gamma_1, gamma_2, phase_offset, b, c, er):
        # bias voltage v_cm
        self._v_bias = v_bias
        # driving voltages
        self._v_A = v_A
        self._v_B = v_B
        # define v_pi as it depends on bias voltage v_cm
        self._v_pi = -np.pi/v_bias
        self._phase_offset = phase_offset
        self._gamma_1 = gamma_1
        self._gamma_2 = gamma_2
        self._b = b
        self._c = c
        self._phase_vl = 0
        self._phase_vr = 0
        self._transmission_vl = 0
        self._transmission_vr = 0
        self._er = er

    @property
    def v_bias(self):
        return self._v_bias

    @property
    def v_A(self):
        return self._v_A

    @property
    def v_B(self):
        return self._v_B

    @property
    def v_pi(self):
        return self._v_pi

    @property
    def phase_offset(self):
        return self._phase_offset

    @property
    def gamma_1(self):
        return self._gamma_1

    @property
    def gamma_2(self):
        return self._gamma_2

    @property
    def b(self):
        return self._b

    @property
    def c(self):
        return self._c

    @property
    def phase_vl(self):
        return self._phase_vl

    @property
    def er(self):
        return self._er

    @phase_vl.setter
    def phase_vl(self, phase_vl):
        self._phase_vl = phase_vl

    @property
    def phase_vr(self):
        return self._phase_vr

    @phase_vr.setter
    def phase_vr(self, phase_vr):
        self._phase_vr = phase_vr

    @property
    def transmission_vl(self):
        return self._transmission_vl

    @transmission_vl.setter
    def transmission_vl(self, transmission_vl):
        self._transmission_vl = transmission_vl

    @property
    def transmission_vr(self):
        return self._transmission_vr

    @transmission_vr.setter
    def transmission_vr(self, transmission_vr):
        self._transmission_vr = transmission_vr

    '''InP Model methods'''

    # define V-dependent intensity tf
    def griffin_intensity_tf(self, v_arm):
        c = self.c
        tf = (1 + np.exp((v_arm - c) / 0.8)) ** -1.25
        return tf

    def griffin_phase(self, v_arm):
        b = self.b
        v_pi = self.v_pi
        v_bias = self.v_bias
        phase = ((2 * b * v_bias * v_pi - np.pi) / v_pi) * v_arm - b * v_arm ** 2
        return phase

    def griffin_eo_tf(self):
        gamma_1 = self.gamma_1
        gamma_2 = self.gamma_2
        phase_offset = self.phase_offset
        v_l = self.v_A
        v_r = self.v_B
        transmission_vl = self.griffin_intensity_tf(v_l)
        transmission_vr = self.griffin_intensity_tf(v_r)
        phase_vl = self.griffin_phase(v_l)
        phase_vr = self.griffin_phase(v_r)
        self.transmission_vl = transmission_vl
        self.transmission_vr = transmission_vr
        self.phase_vl = phase_vl
        self.phase_vr = phase_vr
        ext_r = self.er

        eo_tf_field = np.sqrt(1 - 1 / ext_r) * gamma_1 * gamma_2 * np.sqrt(transmission_vl) * np.exp(phase_vl * 1j) + \
                      np.sqrt(1 - 1 / ext_r) * np.sqrt(np.abs(1 - gamma_1 ** 2) * np.abs(1 - gamma_2 ** 2)) * np.sqrt(
            transmission_vr) * \
                      np.exp((phase_vr + phase_offset) * 1j)
        eo_field_conj = np.conjugate(eo_tf_field)
        eo_tf_power = eo_tf_field * eo_field_conj
        return eo_tf_power

    def griffin_eo_tf_field(self):
        gamma_1 = self.gamma_1
        gamma_2 = self.gamma_2
        phase_offset = self.phase_offset
        v_l = self.v_A
        v_r = self.v_B
        ext_r = self.er
        transmission_vl = self.griffin_intensity_tf(v_l)
        transmission_vr = self.griffin_intensity_tf(v_r)
        phase_vl = self.griffin_phase(v_l)
        phase_vr = self.griffin_phase(v_r)
        self.transmission_vl = transmission_vl
        self.transmission_vr = transmission_vr
        self.phase_vl = phase_vl
        self.phase_vr = phase_vr

        eo_tf_field = np.sqrt(1 - 1 / ext_r) * (gamma_1 * gamma_2 * np.sqrt(transmission_vl) * np.exp(phase_vl * 1j)) + \
                      np.sqrt(1 - 1 / ext_r) * np.sqrt(np.abs(1 - gamma_1 ** 2) * np.abs(1 - gamma_2 ** 2)) * \
                      np.sqrt(transmission_vr) * np.exp((phase_vr + phase_offset) * 1j)
        return eo_tf_field

    def griffin_il_er(self):
        v_l = self.v_A
        v_r = self.v_B
        t_vl = self.griffin_intensity_tf(v_l)
        t_vr = self.griffin_intensity_tf(v_r)
        il = np.sqrt(2) * t_vl
        er = (t_vr + t_vl + 2 * np.sqrt(t_vl * t_vr)) / (t_vr + t_vl - 2 * np.sqrt(t_vl * t_vr))
        er_lin = lin2db(er)
        il_on_er = il / er
        return il, er, il_on_er

    from mzm_model.core.utils import eo_tf_draw, phase_transmission_draw, phase_parametric_draw, \
        transmission_parametric_draw
