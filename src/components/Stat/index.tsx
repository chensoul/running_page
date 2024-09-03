import { intComma } from '@/utils/utils';

interface IStatProperties {
  value: string | number;
  description: string;
  distance: number,
  className?: string;
  citySize?: number;
  onClick?: () => void;
}

const Stat = ({
  value,
  description,
  distance,
  className = 'pb-2 w-full',
  citySize,
  onClick,
}: IStatProperties) => (
  <div className={`${className}`} onClick={onClick}>
    <span className={`text-${citySize || 5}xl font-bold italic`}>
    {intComma(value.toString())}
    </span>
    <span className="text-2xl font-semibold italic">{description}</span>
    { distance > 0 && (<span className="text-5xl font-bold italic">{ " " + distance}</span>)}
    { distance > 0 && (<span className="text-2xl font-semibold italic"> KM</span>)}
  </div>
);

export default Stat;
