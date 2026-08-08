# Claves identificadas como Ingresos Propios Permanentes (IPP).
# Formato: (Cod Subtítulo, Cod Subasignación)
IPP_KEYS = {
    # Subtítulo 03 — Tributos sobre uso de bienes y actividades
    (3, 1001001),  # Patentes municipales de beneficio municipal

    (3, 1002001),  # Derechos de aseo - Impuesto territorial
    (3, 1002002),  # Derechos de aseo - Patentes municipales
    (3, 1002003),  # Derechos de aseo - Cobro directo

    (3, 1003001),  # Urbanización y construcción
    (3, 1003002),  # Permisos provisorios
    (3, 1003003),  # Propaganda
    (3, 1003004),  # Transferencia de vehículos
    (3, 1003999),  # Otros derechos

    (3, 1004001),  # Concesiones

    (3, 2001001),  # Permisos de circulación - Beneficio municipal
    (3, 2002000),  # Licencias de conducir
    (3, 3000000),  # Participación impuesto territorial

    # Subtítulo 05 — Transferencias corrientes
    (5, 3007001),  # Patentes acuícolas

    # Subtítulo 06 — Rentas de la propiedad
    (6, 1000000),  # Arriendo de activos no financieros
    (6, 2000000),  # Dividendos
    (6, 3000000),  # Intereses
    (6, 4000000),  # Participación de utilidades
    (6, 99000000), # Otras rentas de la propiedad

    # Subtítulo 08 — Otros ingresos corrientes
    (8, 2001000),  # Multas de beneficio municipal
    (8, 2001001),  # Multas Ley de Tránsito
    (8, 2001002),  # Multas TAG
    (8, 2001003),  # Multas Decreto 900
    (8, 2001004),  # Registro multas pasajeros infractores
    (8, 2001999),  # Otras multas beneficio municipal
    (8, 2003000),  # Multas Ley de Alcoholes - beneficio municipal
    (8, 2005000),  # Registro multas no pagadas - beneficio municipal
    (8, 2008000),  # Multas e intereses

    # Subtítulo 13 — Transferencias para gastos de capital
    (13, 3005001), # Patentes mineras
    (13, 3005002), # Casinos de juegos
}


# Claves correspondientes a ingresos recibidos desde el
# Fondo Común Municipal (FCM).
FCM_KEYS = {
    (8, 3001000),  # Participación anual
    (8, 3002000),  # Compensaciones FCM
    (8, 3003001),  # Aportes extraordinarios
    (8, 3003002),  # Anticipos por leyes especiales

    # Incorporados con Royalty
    (8, 3006001),  # Fondo Comunas Mineras
    (8, 3006002),  # Fondo Equidad Territorial
}