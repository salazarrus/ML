from vehicleFactory import VehicleFactory
from car import Car

my_car = VehicleFactory.create('car',123)

print('Car id: ' + str(my_car.id))
print('Vehicle maximum velocity: ' + str(my_car.getMaxVelocity()))

bi = VehicleFactory.create('bicycle', 'skladaczek')

print('Bicycle maximum velocity is: ' + str(bi.getMaxVelocity()))

print(Car.__doc__)
print(Car.__name__)
print(Car.__module__)
print(Car.__bases__)
print(Car.__dict__)
